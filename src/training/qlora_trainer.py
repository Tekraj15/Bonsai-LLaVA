# Custom Trainer managing the teacher forward pass
import torch
from transformers import Trainer
from .distillation_loss import DistillationLoss

class BonsaiTrainer(Trainer):
    """
    Custom Trainer for QLoRA Distillation.
    """
    def __init__(self, teacher_model=None, alpha=0.5, temperature=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.distill_loss_fct = DistillationLoss(alpha=alpha, temperature=temperature)
        
        # Move teacher to correct device if needed
        if self.teacher_model:
            self.teacher_model.eval()

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override compute_loss to include distillation.
        """
        # 1. Student Forward
        outputs_student = model(**inputs)
        student_logits = outputs_student.logits
        
        # 2. Teacher Forward (only if teacher is present)
        loss = None
        if self.teacher_model:

            with torch.no_grad():
                outputs_teacher = self.teacher_model(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    attention_mask=inputs.get("attention_mask", None)
                )
                teacher_logits = outputs_teacher.logits
            
            # 3. Compute Distillation Loss
            loss = self.distill_loss_fct(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=inputs["labels"]
            )
        else:
            # Fallback to standard loss if no teacher (e.g. pure SFT)
            loss = outputs_student.loss
            
        return (loss, outputs_student) if return_outputs else loss
