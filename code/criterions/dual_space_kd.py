import math
import torch
from .various_divergence import VariousDivergence
from .soft_dtw_cuda import SoftDTW


class DualSpaceKD(VariousDivergence):
    def __init__(self, args, padding_id=-100) -> None:
        super().__init__(args, padding_id=padding_id)
        if self.dtw_rate > 0:
            self.dtw = SoftDTW(use_cuda=True, gamma=args.dtw_gamma)
        else:
            self.dtw = None

        self.dtw_gamma_start = getattr(args, "dtw_gamma_start", getattr(args, "dtw_gamma", 2.0))
        self.dtw_gamma_end = getattr(args, "dtw_gamma_end", 0.8)
        self.dtw_gamma_steps = getattr(args, "dtw_gamma_steps", 0)

        self._global_step = 0
        self.kd_warmup_steps = getattr(args, "kd_warmup_steps", 300)
        self.dtw_warmup_steps = getattr(args, "dtw_warmup_steps", 0)

    def forward(
        self, 
        distiller, 
        input_data, 
        output_data, 
        logging_output, 
        batch_denom, 
    ):
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        self.distiller = distiller
        outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            position_ids=input_data.get("position_ids", None), 
            output_hidden_states=True
        )
        logits = outputs.logits
        log = {}
        student_probs = torch.softmax(logits, -1, dtype=torch.float32)
        student_entropy = -(student_probs * torch.log(student_probs + 1e-9)).sum(-1)
        pad_mask = output_data["label"].ne(self.padding_id)
        entropy_weights = student_entropy.detach() * pad_mask.float()
        valid_tokens = pad_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        entropy_sum = entropy_weights.sum(dim=1, keepdim=True) + 1e-9
        entropy_weights = entropy_weights * (valid_tokens / entropy_sum)

        ce_loss = self.compute_cross_entropy_loss(
            outputs.logits, output_data["label"], log=log
        )[0]

        with torch.no_grad():
            teacher_model.eval()
            teacher_outputs = teacher_model(
                input_data[f"teacher_{distiller.teacher_model_type}_input_ids"],
                attention_mask=input_data[f"teacher_{distiller.teacher_model_type}_attention_mask"],
                position_ids=input_data.get(f"teacher_{distiller.teacher_model_type}_position_ids", None), 
                output_hidden_states=True)
        
        kd_loss, log = self.compute_dual_space_kd_loss(
            outputs, teacher_outputs, output_data, distiller, log
        )

        if self.dtw_rate > 0 and self.dtw_gamma_steps and self.dtw_gamma_steps > 0 and self.dtw is not None:
            progress = min(1.0, float(self._global_step + 1) / float(self.dtw_gamma_steps))
            current_gamma = self.dtw_gamma_start + (self.dtw_gamma_end - self.dtw_gamma_start) * progress
            self.dtw.gamma = current_gamma

        dtw_loss, log = self.compute_dtw_loss(
            outputs, teacher_outputs, input_data, output_data, distiller, log
        )

        if self.dtw_warmup_steps and self.dtw_warmup_steps > 0:
            dtw_warmup_scale = min(1.0, float(self._global_step + 1) / float(self.dtw_warmup_steps))
        else:
            dtw_warmup_scale = 1.0

        weighted_ce_loss = ce_loss
        weighted_kd_loss = kd_loss
        weighted_dtw_loss = dtw_loss * dtw_warmup_scale

        loss = self.ce_rate * weighted_ce_loss + self.kd_rate * weighted_kd_loss + self.dtw_rate * weighted_dtw_loss

        log["loss"] = loss
        log["weighted_ce_loss"] = weighted_ce_loss
        log["weighted_kd_loss"] = weighted_kd_loss
        log["weighted_dtw_loss"] = weighted_dtw_loss
        log["student_entropy_mean"] = entropy_weights.mean()
        log["student_entropy_std"] = entropy_weights.std()

        accuracy = self.compute_token_accuracy(
            logits, output_data["label"], 
        )
        log["accuracy"] = accuracy

        logging_output = self.record_logging_output(
            logging_output, batch_denom, log
        )
        self._global_step += 1
        return loss / batch_denom, logging_output

    def compute_dual_space_kd_loss(
        self, outputs, teacher_outputs, output_data, distiller, log
    ):
        target = output_data["label"]
        pad_mask = target.ne(self.padding_id)
        teacher_target = output_data[f"teacher_{distiller.teacher_model_type}_label"]
        teacher_pad_mask = teacher_target.ne(self.padding_id)

        stu_probs = torch.softmax(outputs.logits, -1, dtype=torch.float32)
        stu_entropy = -(stu_probs * torch.log(stu_probs + 1e-9)).sum(-1)
        w_s = stu_entropy.detach() * pad_mask.float()
        s_tokens = pad_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        w_s_sum = w_s.sum(dim=1, keepdim=True) + 1e-9
        w_s = w_s * (s_tokens / w_s_sum)

        tea_probs = torch.softmax(teacher_outputs.logits, -1, dtype=torch.float32)
        tea_entropy = -(tea_probs * torch.log(tea_probs + 1e-9)).sum(-1)
        max_entropy = math.log(teacher_outputs.logits.size(-1))
        tea_certainty = (1.0 - (tea_entropy / (max_entropy + 1e-9))).clamp(min=0.0, max=1.0)
        w_t = tea_certainty.detach() * teacher_pad_mask.float()
        t_tokens = teacher_pad_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        w_t_sum = w_t.sum(dim=1, keepdim=True) + 1e-9
        w_t = w_t * (t_tokens / w_t_sum)

        hiddens = outputs.hidden_states[-1]
        teacher_hiddens = teacher_outputs.hidden_states[-1]

        # student space
        t2s_hiddens = distiller.projectors["t2s"](teacher_hiddens)
        t2s_logits = t2s_hiddens.matmul(
            distiller.student_model.lm_head.weight.detach().transpose(-1, -2)
        )
        t2s_ce_loss = self.compute_cross_entropy_loss(t2s_logits, target)[0]
        t2s_ce_loss_weighted = t2s_ce_loss

        t2s_kd_vec = self.dist_func(
            outputs.logits, t2s_logits.detach(), target, reduction="none", use_tea_temp=True
        )
        t2s_probs = torch.softmax(t2s_logits.detach(), -1, dtype=torch.float32)
        t2s_conf = t2s_probs.max(dim=-1)[0]
        kd_gate = t2s_conf
        if self.kd_warmup_steps and self.kd_warmup_steps > 0:
            warmup_scale = min(1.0, float(self._global_step + 1) / float(self.kd_warmup_steps))
            kd_gate = kd_gate * warmup_scale
        kd_gate = kd_gate * pad_mask.float()
        t2s_kd_loss = (t2s_kd_vec * kd_gate).sum()
        t2s_kd_loss_weighted = (t2s_kd_vec * kd_gate * w_s).sum()

        # teacher space
        s2t_hiddens = distiller.projectors["s2t"](hiddens)
        s2t_logits = distiller.teacher_model.lm_head(s2t_hiddens)
        s2t_kd_vec = self.compute_forward_kl_divergence(
            s2t_logits, teacher_outputs.logits, teacher_target, reduction="none"
        )
        s2t_kd_loss = (s2t_kd_vec * teacher_pad_mask.float()).sum()
        s2t_kd_loss_weighted = (s2t_kd_vec * teacher_pad_mask.float() * w_t).sum()

        kd_loss = t2s_ce_loss_weighted + t2s_kd_loss_weighted + s2t_kd_loss_weighted

        t2s_acc = (t2s_logits.argmax(-1).eq(target) * pad_mask).sum()
        s2t_acc = (s2t_logits.argmax(-1).eq(teacher_target) * teacher_pad_mask).sum() * pad_mask.sum() / teacher_pad_mask.sum()

        log["t2s_ce_loss"] = t2s_ce_loss
        log["t2s_kd_loss"] = t2s_kd_loss
        log["s2t_kd_loss"] = s2t_kd_loss
        log["t2s_acc"] = t2s_acc
        log["s2t_acc"] = s2t_acc
        log["kd_loss"] = kd_loss
        return kd_loss, log
    
    def compute_dtw_loss(self, outputs, teacher_outputs, input_data, output_data, distiller, log):
        if self.dtw_rate == 0 or self.dtw is None:
            log["dtw_loss"] = 0.0
            return torch.tensor(0.0, device=outputs.logits.device), log

        pad_mask = output_data["label"].ne(self.padding_id)
        teacher_pad_mask = output_data[f"teacher_{distiller.teacher_model_type}_label"].ne(self.padding_id)

        stu_target_embeds, tea_target_embeds = self._get_target_embeddings(
            distiller, input_data, output_data, pad_mask, teacher_pad_mask
        )

        hiddens = outputs.hidden_states[-1]
        teacher_hiddens = teacher_outputs.hidden_states[-1]

        dtw_embed_projector = distiller.projectors["dtw_embed_t2s"]
        projected_teacher_embeds = dtw_embed_projector(tea_target_embeds)
        loss_embed = self._calculate_alignment_loss(
            stu_target_embeds, projected_teacher_embeds, pad_mask, teacher_pad_mask
        )

        projected_teacher_hiddens = distiller.projectors["t2s"](teacher_hiddens)
        loss_hidden = self._calculate_alignment_loss(
            hiddens, projected_teacher_hiddens, pad_mask, teacher_pad_mask
        )

        total_dtw_loss = loss_hidden + loss_embed

        log["dtw_loss"] = total_dtw_loss.item()
        log["dtw_hidden_loss"] = loss_hidden.item()
        log["dtw_embed_loss"] = loss_embed.item()

        return total_dtw_loss, log

    def _calculate_alignment_loss(self, student_embs, teacher_embs, student_mask, teacher_mask):
        batch_size = student_embs.size(0)
        total_loss = torch.tensor(0.0, device=student_embs.device, requires_grad=True)
        non_empty_pairs = 0

        for i in range(batch_size):
            s_len = student_mask[i].sum().item()
            t_len = teacher_mask[i].sum().item()

            if s_len == 0 or t_len == 0:
                continue

            non_empty_pairs += 1

            s_seq = student_embs[i, :s_len, :]
            t_seq = teacher_embs[i, :t_len, :]

            c_stu_tea = 1.0 - torch.cosine_similarity(
                s_seq.unsqueeze(1), t_seq.unsqueeze(0), dim=-1
            )

            c_stu_stu = 1.0 - torch.cosine_similarity(
                s_seq.unsqueeze(1), s_seq.unsqueeze(0), dim=-1
            )

            c_tea_tea = 1.0 - torch.cosine_similarity(
                t_seq.unsqueeze(1), t_seq.unsqueeze(0), dim=-1
            )

            s2t = self.dtw.forward_with_cost_matrix(c_stu_tea.unsqueeze(0))
            s2s = self.dtw.forward_with_cost_matrix(c_stu_stu.unsqueeze(0))
            t2t = self.dtw.forward_with_cost_matrix(c_tea_tea.unsqueeze(0))

            pair_loss = s2t - 0.5 * (s2s + t2t)

            total_loss = total_loss + pair_loss.squeeze()

        if non_empty_pairs == 0:
            return torch.tensor(0.0, device=student_embs.device)

        return total_loss

    def _get_target_embeddings(self, distiller, input_data, output_data, pad_mask, teacher_pad_mask):
        target = output_data["label"]
        teacher_target = output_data[f"teacher_{distiller.teacher_model_type}_label"]

        if hasattr(distiller.student_model, "model") \
            and hasattr(distiller.student_model.model, "embed_tokens"):
            stu_embed_tokens = distiller.student_model.model.embed_tokens
        elif hasattr(distiller.student_model, "model") \
            and hasattr(distiller.student_model.model, "model") \
            and hasattr(distiller.student_model.model.model, "embed_tokens"):
            stu_embed_tokens = distiller.student_model.model.model.embed_tokens
        elif hasattr(distiller.student_model, "transformer") \
            and hasattr(distiller.student_model.transformer, "wte"):
            stu_embed_tokens = distiller.student_model.transformer.wte
        else:
            raise NotImplementedError

        if hasattr(distiller.teacher_model, "model") \
            and hasattr(distiller.teacher_model.model, "embed_tokens"):
            tea_embed_tokens = distiller.teacher_model.model.embed_tokens
        elif hasattr(distiller.teacher_model, "model") \
            and hasattr(distiller.teacher_model.model, "model") \
            and hasattr(distiller.teacher_model.model.model, "embed_tokens"):
            tea_embed_tokens = distiller.teacher_model.model.model.embed_tokens
        elif hasattr(distiller.teacher_model, "transformer") \
            and hasattr(distiller.teacher_model.transformer, "wte"):
            tea_embed_tokens = distiller.teacher_model.transformer.wte
        else:
            raise NotImplementedError

        formal_target = torch.where(pad_mask, target, torch.zeros_like(target))
        stu_target_embeds = stu_embed_tokens(formal_target).detach()

        formal_teacher_target = torch.where(teacher_pad_mask, teacher_target, torch.zeros_like(teacher_target))
        tea_target_embeds = tea_embed_tokens(formal_teacher_target).detach()

        return stu_target_embeds, tea_target_embeds
