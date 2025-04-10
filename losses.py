import torch
import torch.distributed.nn
import torch.nn as nn
import torch.nn.functional as F

from utils import general_utils


class CLIPLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.first_time = True
        self.cont = args.cont  # Weight for contrastive loss, as in equation 7 in the paper
        self.sneg = args.sneg  # Weight for hard negatives loss, as in equation 7 in the paper
        self.uni = args.uni  # Weight for uni-modal loss, as in equation 7 in the paper
        self.args = args

    def l2(self, out, targets, reduction='mean'):
        """
        Compute squared L2 loss (the embeddings are normalized in the model).
        :param out: Predicted embeddings (batch_size, embedding_size)
        :param targets: Target embeddings (batch_size, embedding_size)
        :param reduction: 'mean' or 'sum' reduction method
        :return: L2 loss
        """
        # squared l2 - it does not divide by the latent dimension
        # should have shape (batch_size, embedding_size)
        assert out.shape == targets.shape, f'{out.shape} != {targets.shape}'
        assert out.shape[0] > 1

        squared_error = F.mse_loss(out, targets, reduction='none')
        return squared_error.sum(dim=1).mean() if reduction == 'mean' else squared_error.sum(dim=1)

    def compute_uni_modal_loss(self, text_embed, num_img_per_gpu):
        """
        Compute uni-modal loss when multiple positive captions exist per image.
        :param text_embed: Text embeddings (num_txt_per_gpu, embedding_dim)
        :param num_img_per_gpu: Number of images per GPU
        :return: Uni-modal loss
        """
        if self.num_pos > 1:
            pos1_embed = text_embed[:num_img_per_gpu]
            pos2_embed = text_embed[num_img_per_gpu:2 * num_img_per_gpu]
            return self.l2(out=pos1_embed, targets=pos2_embed, reduction='mean')
        raise ValueError("Uni-modal loss requires multiple positive samples")

    def _calc_num_pos(self):
        """
        Calculate the number of positive captions per image, in the hard negatives iteration
        """
        # adding the additional positives: p_3, p_4 in the paper
        self.num_pos = self.args.additional_positives + 1

        # adding the shuffled positives: p_2 in the paper
        if self.args.shuffled_positive:
            self.num_pos += 1  # account for shuffled positives

    def _initialize_indices_and_labels(self, num_img_per_gpu, device):
        """Initialize indices and labels for loss computation."""
        # Calculate the number of positive captions per image for hard negatives iteration
        self._calc_num_pos()

        # Basic indices for the current GPU batch, Shape: (num_img_per_gpu,)
        self.indices = torch.arange(num_img_per_gpu, device=device)

        # Labels for standard contrastive loss (without negatives), Shape: (num_img_per_gpu,)
        self.labels_without_neg = (num_img_per_gpu * general_utils.get_rank() + self.indices)

        # Index where negatives start within each GPU's data
        self.gpu_negatives_start_idx = num_img_per_gpu * self.num_pos

        # Target labels for separate negatives approach (all zeros since this would be the positive samples)
        # Shape: (num_img_per_gpu,)
        self.separate_negatives_targets = torch.zeros(num_img_per_gpu, dtype=torch.long, device=device)

        # Labels when using both positives and negatives in the same batch
        # Shape: (num_img_per_gpu,)
        self.labels_with_neg = (num_img_per_gpu * 2 * general_utils.get_rank() + self.indices)

        # Mark initialization as complete
        self.first_time = False

    def _compute_standard_clip_loss(self, logits_per_image, logits_per_text, num_img_per_gpu):
        """
        Compute standard CLIP contrastive loss.

        This is the standard CLIP loss computation when the number of images and texts are equal.
        """
        # Standard contrastive loss calculation
        img_to_txt_loss = F.cross_entropy(logits_per_image, self.labels_without_neg)
        txt_to_img_loss = F.cross_entropy(logits_per_text, self.labels_without_neg)
        contrastive_loss = (img_to_txt_loss + txt_to_img_loss) / 2

        # Calculate accuracy
        with torch.no_grad():
            pred = torch.argmax(logits_per_image, dim=-1)
            correct = pred.eq(self.labels_without_neg).sum()
            acc = 100 * correct / num_img_per_gpu

        metrics = {
            "combined_loss": contrastive_loss,
            "contrastive_loss": contrastive_loss,
            "img_to_txt_loss": img_to_txt_loss,
            "txt_to_img_loss": txt_to_img_loss,
            "clip_acc": acc,
        }

        return metrics

    def _compute_accuracy_with_hard_negatives(self, logits_per_image_list, neg_logits_list, num_img_per_gpu):
        """
        Compute accuracy metrics for the case of hard negatives or multiple positives.
        """
        with torch.no_grad():
            correct_list = []

            for i, logits in enumerate(logits_per_image_list):
                pred = torch.argmax(logits, dim=-1)

                if self.args.hard_negatives_separate:
                    # For separate negatives: prediction need to be correct in both contrastive and hard negative parts
                    correct_in_contrastive_loss = pred.eq(self.labels_without_neg)
                    correct_in_hard_negatives = torch.argmax(neg_logits_list[i], dim=-1) == 0
                    correct_list.append((correct_in_contrastive_loss & correct_in_hard_negatives).sum())
                else:
                    # For unified negatives: check against labels that include negatives
                    correct_list.append(pred.eq(self.labels_with_neg).sum())

            # Average accuracy across all positive sets
            acc = 100 * (sum(correct_list) / self.num_pos) / num_img_per_gpu

            return acc

    def _compute_loss_with_hard_negatives(self, logits_per_image, logits_per_text, text_embed,
                                          num_img_per_gpu, num_txt_per_gpu, num_txt_across_gpu, device):
        """
        Compute loss using hard negatives or multiple positive samples.
        This implements the equations 4,5,6,7 from the paper
        """
        img_to_txt_loss_list, txt_to_img_loss_list, neg_loss_list = [], [], []
        logits_per_image_list, neg_logits_list = [], []

        # Create indices for all text samples across GPUs
        all_idx = torch.arange(num_txt_across_gpu, device=device)  # Shape: (num_txt_across_gpu,)

        # Identify hard negative samples, Shape: (num_txt_across_gpu,)
        indices_hard_negative = all_idx % num_txt_per_gpu >= self.gpu_negatives_start_idx

        # Get starting index for the current GPU's data
        idx_current_gpu = num_txt_per_gpu * general_utils.get_rank()

        # Extract logits for hard negatives from the current GPU, Shape: (num_img_per_gpu,)
        neg_logits = logits_per_image[self.indices, self.indices + self.gpu_negatives_start_idx + idx_current_gpu]

        for i in range(self.num_pos):
            # Calculate index range for the current positive set
            start_idx = i * num_img_per_gpu
            end_idx = start_idx + num_img_per_gpu

            # Identify indices for the current positive set across all GPUs, Shape: (num_txt_across_gpu,)
            indices_next_pos = (all_idx % num_txt_per_gpu >= start_idx) & (all_idx % num_txt_per_gpu < end_idx)

            # Separate negatives approach (Equation 3 in paper)
            if self.args.hard_negatives_separate:
                # Compute standard contrastive loss (Equation 4 in paper)
                logits_per_image_list.append(logits_per_image[:, indices_next_pos])
                img_to_txt_loss_list.append(F.cross_entropy(logits_per_image_list[-1], self.labels_without_neg))

                # Compute hard negative loss (Equation 5 in paper)
                # Stack positive and negative logits for binary classification
                curr_pos_logits = logits_per_image[self.indices, self.indices + start_idx + idx_current_gpu]
                neg_logits_list.append(torch.stack([curr_pos_logits, neg_logits], dim=1))
                neg_loss_list.append(F.cross_entropy(neg_logits_list[-1], self.separate_negatives_targets))
            else:
                # Unified loss with hard negatives (Equation 2 in paper, adapted for multiple positives)
                indices_next_positive_and_negative = indices_next_pos | indices_hard_negative
                logits_per_image_list.append(logits_per_image[:, indices_next_positive_and_negative])
                img_to_txt_loss_list.append(F.cross_entropy(logits_per_image_list[-1], self.labels_with_neg))

            # Compute text-to-image loss for current positive set
            txt_to_img_loss_list.append(F.cross_entropy(logits_per_text[start_idx:end_idx], self.labels_without_neg))

        # Average losses across all positive sets
        img_to_txt_loss = sum(img_to_txt_loss_list) / self.num_pos
        txt_to_img_loss = sum(txt_to_img_loss_list) / self.num_pos
        contrastive_loss = (img_to_txt_loss + txt_to_img_loss) / 2

        combined_loss = contrastive_loss
        metrics = {
            "contrastive_loss": contrastive_loss,
            "img_to_txt_loss": img_to_txt_loss,
            "txt_to_img_loss": txt_to_img_loss,
        }

        # Add separate negative loss if configured (Equation 5 in paper)
        if self.args.hard_negatives_separate:
            sneg_loss = sum(neg_loss_list) / self.num_pos
            combined_loss = self.cont * contrastive_loss + self.sneg * sneg_loss
            metrics["sneg_loss"] = sneg_loss

        # Add uni-modal loss if configured (Equation 6 in paper)
        if self.args.uni_modal_loss:
            uni_modal_loss = self.compute_uni_modal_loss(text_embed, num_img_per_gpu)
            combined_loss += self.uni * uni_modal_loss
            metrics["uni_modal_loss"] = uni_modal_loss

        # Compute accuracy metrics
        acc = self._compute_accuracy_with_hard_negatives(logits_per_image_list, neg_logits_list, num_img_per_gpu)
        metrics["clip_acc"] = acc
        metrics["combined_loss"] = combined_loss
        return metrics

    def forward(self, outputs):
        """
        :param outputs: Dictionary containing model outputs with the following keys:
        - "logits_per_image": Logits for image-to-text matching
        - "logits_per_text": Logits for text-to-image matching
        - "text_only": Text embeddings
        - "cos-sim": Cosine similarity between image and text embeddings
        :return: Dictionary containing loss metrics.
        """

        logits_per_image = outputs["logits_per_image"]  # Shape: (num_img_per_gpu, num_txt_per_gpu * num_gpus)
        logits_per_text = outputs["logits_per_text"]  # Shape: (num_txt_per_gpu, num_img_per_gpu * num_gpus)
        text_embed = outputs["text_only"]  # Shape: (num_txt_per_gpu, text_embed_dim)
        cos_sim = outputs['cos-sim']

        # Get batch dimensions
        num_img_per_gpu, num_txt_per_gpu = logits_per_image.size(0), logits_per_text.size(0)
        num_txt_across_gpu = logits_per_image.size(1)
        device = logits_per_image.device

        # Initialize indices and labels if this is the first call
        if self.first_time:
            self._initialize_indices_and_labels(num_img_per_gpu, device)

        # Handle two cases: standard CLIP loss (equal images/texts) or with additional positives/negatives
        if num_img_per_gpu != num_txt_per_gpu:
            # Case: Using hard negatives or additional positives
            metrics = self._compute_loss_with_hard_negatives(logits_per_image, logits_per_text, text_embed,
                                                             num_img_per_gpu, num_txt_per_gpu, num_txt_across_gpu,
                                                             device)
        else:
            # Case: Standard CLIP loss computation
            metrics = self._compute_standard_clip_loss(logits_per_image, logits_per_text, num_img_per_gpu)

        metrics['cos-sim'] = cos_sim

        return metrics
