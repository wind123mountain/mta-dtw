import math
import torch
from .various_divergence import VariousDivergence
from .soft_dtw_cuda import SoftDTW
from utils import log_rank
import re


def align_sequences(tea_seq, stu_seq, student_tokenizer, teacher_tokenizer):
    i, j = 0, 0
    t2s_align, s2t_align = [], []
    history_tea_seq, history_stu_seq = "", ""
    
    tea_seq = [token.replace('▁', '').replace('Ġ', '') for token in tea_seq]
    stu_seq = [token.replace('▁', '').replace('Ġ', '') for token in stu_seq]

    while i < len(tea_seq) and j < len(stu_seq):
        if history_tea_seq == history_stu_seq and (
            tea_seq[i] == stu_seq[j] or (
                tea_seq[i] == teacher_tokenizer.eos_token and \
                stu_seq[j] == student_tokenizer.eos_token
            )
        ):
            history_tea_seq += tea_seq[i]
            history_stu_seq += stu_seq[j]
            t2s_align.append(i)
            s2t_align.append(j)
            i += 1
            j += 1
        elif len(history_tea_seq) > len(history_stu_seq):
            history_stu_seq += stu_seq[j]
            j += 1
        elif len(history_tea_seq) < len(history_stu_seq):
            history_tea_seq += tea_seq[i]
            i += 1
        else:
            history_tea_seq += tea_seq[i]
            history_stu_seq += stu_seq[j]
            i += 1
            j += 1
            
    return t2s_align, s2t_align


class DualSpaceKDV2WithETA(VariousDivergence):
    def __init__(self, args, padding_id=-100) -> None:
        super().__init__(args, padding_id=padding_id)
        if self.dtw_rate > 0:
            gamma = getattr(args, "dtw_gamma", 2.0)
            self.dtw = SoftDTW(use_cuda=True, gamma=gamma)
        self.dtw_gamma_start = getattr(args, "dtw_gamma_start", getattr(args, "dtw_gamma", 2.0))
        self.dtw_gamma_end = getattr(args, "dtw_gamma_end", 0.8)
        self.dtw_gamma_steps = getattr(args, "dtw_gamma_steps", 3570)
        self.dtw_band_width = getattr(args, "dtw_band_width", 5)
        self.dtw_band_penalty = getattr(args, "dtw_band_penalty", 1.0)
        self.dtw_band_center_blend = getattr(args, "dtw_band_center_blend", 0.7)
        self.dtw_band_entropy_coef = getattr(args, "dtw_band_entropy_coef", 2.0)
        self.dtw_band_warmup_steps = getattr(args, "dtw_band_warmup_steps", 0)
        self.dtw_warmup_steps = getattr(args, "dtw_warmup_steps", 0)
        self.dtw_band_source = getattr(args, "dtw_band_source", "cma")
        self._global_step = 0
        self.last_align = None
        self.last_pad_mask = None
        self.last_teacher_pad_mask = None

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
        teacher_model.eval()

        self.distiller = distiller
        outputs = model(
            input_data["input_ids"],
            attention_mask=input_data["attention_mask"],
            position_ids=input_data.get("position_ids", None), 
            output_hidden_states=True
        )
        logits = outputs.logits
        log = {}
        ce_loss = self.compute_cross_entropy_loss(
            outputs.logits, output_data["label"], log=log
        )[0] / batch_denom
        log["nll_loss"] = ce_loss

        # Construct batch dict for compatibility with existing methods
        teacher_label_key = f"teacher_{distiller.teacher_model_type}_label"
        # Create label_batch dict with loss_denom
        label_batch = dict(output_data)
        label_batch["loss_denom"] = batch_denom
        batch = {
            "input_batch": input_data,
            "label_batch": label_batch,
            "teacher_input_batch": {
                f"teacher_{distiller.teacher_model_type}_input_ids": input_data.get(f"teacher_{distiller.teacher_model_type}_input_ids"),
                f"teacher_{distiller.teacher_model_type}_attention_mask": input_data.get(f"teacher_{distiller.teacher_model_type}_attention_mask"),
            },
            "teacher_label_batch": {
                "label": output_data.get(teacher_label_key, output_data["label"])
            }
        }
        if f"teacher_{distiller.teacher_model_type}_position_ids" in input_data:
            batch["teacher_input_batch"][f"teacher_{distiller.teacher_model_type}_position_ids"] = input_data[f"teacher_{distiller.teacher_model_type}_position_ids"]
        
        kd_student_label_batch = batch["label_batch"]
        kd_teacher_label_batch = batch["teacher_label_batch"]
        if "op_input_batch" in input_data:     # on-policy scenario
            batch["op_input_batch"] = input_data["op_input_batch"]
            batch["op_label_batch"] = output_data["op_label_batch"]
            batch["op_teacher_input_batch"] = input_data["op_teacher_input_batch"]
            batch["op_teacher_label_batch"] = output_data["op_teacher_label_batch"]
            outputs = model(**batch["op_input_batch"], output_hidden_states=True)
            kd_student_label_batch = batch["op_label_batch"]
            kd_teacher_label_batch = batch["op_teacher_label_batch"]
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    **batch["op_teacher_input_batch"], 
                    output_hidden_states=True
                )
            kd_loss, log = self.compute_on_policy_dual_space_kd_loss_with_eta(
                outputs, teacher_outputs, batch, distiller, log
            )
        else:    # off-policy scenario
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    **batch["teacher_input_batch"], 
                    output_hidden_states=True
                )
            kd_loss, log = self.compute_dual_space_kd_loss_with_eta(
                outputs, teacher_outputs, batch, distiller, log
            )

        dtw_loss = logits.new_zeros(())
        if self.dtw_rate > 0:
            if self.dtw_gamma_steps and self.dtw_gamma_steps > 0:
                progress = min(1.0, float(self._global_step + 1) / float(self.dtw_gamma_steps))
                current_gamma = self.dtw_gamma_start + (self.dtw_gamma_end - self.dtw_gamma_start) * progress
                self.dtw.gamma = current_gamma
            dtw_loss, log = self.compute_dtw_loss(
                outputs,
                teacher_outputs,
                distiller,
                kd_student_label_batch["label"],
                kd_teacher_label_batch["label"],
                log,
            )

        weighted_ce_loss = ce_loss
        weighted_kd_loss = kd_loss
        if self.dtw_warmup_steps and self.dtw_warmup_steps > 0:
            dtw_warmup_scale = min(1.0, float(self._global_step + 1) / float(self.dtw_warmup_steps))
        else:
            dtw_warmup_scale = 1.0
        weighted_dtw_loss = dtw_loss * dtw_warmup_scale
        loss = self.ce_rate * weighted_ce_loss + self.kd_rate * weighted_kd_loss + self.dtw_rate * weighted_dtw_loss
        log["loss"] = loss
        log["weighted_ce_loss"] = weighted_ce_loss
        log["weighted_kd_loss"] = weighted_kd_loss
        log["weighted_dtw_loss"] = weighted_dtw_loss

        accuracy = self.compute_token_accuracy(logits, output_data["label"])
        log["accuracy"] = accuracy

        logging_output = self.record_logging_output(logging_output, log)
        self._global_step += 1
        return loss, logging_output
    

    def compute_dual_space_kd_loss_with_eta(
        self, outputs, teacher_outputs, batch, distiller, log
    ):
        target = batch["label_batch"]["label"]
        teacher_target = batch["teacher_label_batch"]["label"]

        pad_mask = target.ne(self.padding_id)
        teacher_pad_mask = teacher_target.ne(self.padding_id)

        hiddens = outputs.hidden_states[-1]
        teacher_hiddens = teacher_outputs.hidden_states[-1]

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
        
        # teacher space
        if self.args.init_s2t_projector:
            stu_lm_head = distiller.student_model.lm_head.weight.detach().transpose(0, 1)
            stu_lm_head = stu_lm_head[:, distiller.student_overlap_token_ids]
            if self.args.topk_vocab != -1:
                stu_lm_head = stu_lm_head[:, :self.args.topk_vocab]
            s2t_projector = stu_lm_head @ distiller.part_teacher_head_pinv
            stu_v_hiddens = hiddens @ s2t_projector
        else:
            stu_v_hiddens = distiller.projectors["s2t"](hiddens)

        tea_v_hiddens = distiller.projectors["t2s"](teacher_hiddens)  # m x D x D x d -> m x d

        t_preds = teacher_outputs.logits.argmax(-1)
        t_preds = torch.where(teacher_pad_mask, t_preds, teacher_target)

        t_preds_as_label = []
        t2s_hiddens_align = torch.zeros_like(tea_v_hiddens).to(tea_v_hiddens.device).to(tea_v_hiddens)   # m x d -> n x d    在s的维度上是t的hiddens
        s2t_hiddens_align = torch.zeros_like(stu_v_hiddens).to(stu_v_hiddens.device).to(stu_v_hiddens)   # n x D -> m x D    在t的维度上是s的hiddens
        align_ratio = []
        align_cache = torch.zeros(
            target.size(0),
            target.size(1),
            teacher_target.size(1),
            device=target.device,
            dtype=tea_v_hiddens.dtype,
        )
        for i in range(t_preds.shape[0]):
            indices_t_preds = torch.where(t_preds[i] != -100)[0]
            indices_t_target = torch.where(teacher_target[i] != -100)[0]
            indices_s_target = torch.where(target[i] != -100)[0]

            if indices_t_preds.shape[0] == 0:
                t_preds_as_label.append(torch.tensor([-100]*t_preds.shape[-1], device=tea_v_hiddens.device))
                align_ratio.append(1.0)
                continue
            
            cur_t_preds = t_preds[i][indices_t_preds[0]: indices_t_preds[-1]+1]
            cur_t_target = teacher_target[i][indices_t_target[0]: indices_t_target[-1]+1]
            cur_t_target_tokens = distiller.teacher_tokenizer.convert_ids_to_tokens(cur_t_target)

            cur_s_target = target[i][indices_s_target[0]: indices_s_target[-1]+1]
            cur_s_target_tokens = distiller.student_tokenizer.convert_ids_to_tokens(cur_s_target)

            align_t_idx, align_s_idx = align_sequences(
                cur_t_target_tokens, 
                cur_s_target_tokens, 
                distiller.student_tokenizer, 
                distiller.teacher_tokenizer
            )
            if align_t_idx == [] and align_s_idx == []:
                t_preds_as_label.append(torch.tensor([-100] * t_preds.shape[-1], device=tea_v_hiddens.device))
                align_ratio.append(1.0)
                continue
            cur_align_ratio = len(align_s_idx) / len(cur_s_target)
            align_ratio.append(cur_align_ratio)

            cur_t_preds_as_label_1 = target[i][:indices_s_target[0]].cpu().tolist()
            cur_t_preds_as_label_2 = [-100] * len(cur_s_target)
            for _t_idx, _s_idx in zip(align_t_idx, align_s_idx):
                tmp_t_token = distiller.teacher_tokenizer.convert_ids_to_tokens([cur_t_preds[_t_idx]])
                if cur_t_preds[_t_idx] == distiller.teacher_tokenizer.eos_token_id:
                    cur_t_preds_as_label_2[_s_idx] = distiller.student_tokenizer.eos_token_id
                    s_abs_idx = _s_idx + indices_s_target[0]
                    t_abs_idx = _t_idx + indices_t_target[0]
                    s2t_hiddens_align[i, t_abs_idx, :] = stu_v_hiddens[i, s_abs_idx]
                    t2s_hiddens_align[i, s_abs_idx, :] = tea_v_hiddens[i, t_abs_idx]
                    align_cache[i, s_abs_idx, t_abs_idx] = 1.0
                else:
                    try:
                        tmp = distiller.student_tokenizer.convert_tokens_to_ids(tmp_t_token)
                        if len(tmp) == 1 and tmp[0] is not None:
                            cur_t_preds_as_label_2[_s_idx] = tmp[0]
                            s_abs_idx = _s_idx + indices_s_target[0]
                            t_abs_idx = _t_idx + indices_t_target[0]
                            s2t_hiddens_align[i, t_abs_idx, :] = stu_v_hiddens[i, s_abs_idx]
                            t2s_hiddens_align[i, s_abs_idx, :] = tea_v_hiddens[i, t_abs_idx]
                            align_cache[i, s_abs_idx, t_abs_idx] = 1.0
                        else:
                            cur_t_preds_as_label_2[_s_idx] = -100
                    except:
                        cur_t_preds_as_label_2[_s_idx] = -100

            assert len(cur_t_preds_as_label_2) == len(cur_s_target)
            cur_t_preds_as_label_3 = target[i][indices_s_target[-1]+1:].cpu().tolist()
            
            cur_t_preds_as_label = cur_t_preds_as_label_1 + cur_t_preds_as_label_2 + cur_t_preds_as_label_3
            cur_t_preds_as_label = cur_t_preds_as_label[:teacher_target.shape[1]]
            t_preds_as_label.append(torch.tensor(cur_t_preds_as_label, device=teacher_target.device))

        t_preds_as_label = torch.cat(t_preds_as_label, dim=0).reshape(-1, teacher_target.shape[1])
        log["align_ratio"] = torch.tensor(align_ratio, device=teacher_target.device).mean()
        if self.dtw_rate > 0:
            self.last_align = align_cache.detach()
            self.last_pad_mask = pad_mask.detach()
            self.last_teacher_pad_mask = teacher_pad_mask.detach()
        else:
            self.last_align = None
            self.last_pad_mask = None
            self.last_teacher_pad_mask = None
        t2s_logits = t2s_hiddens_align.matmul(
            distiller.student_model.lm_head.weight.detach().transpose(-1, -2)
        )  # n x d x d x V_stu -> n x V_stu  [bsz x seq-len x V_stu]

        stu_align_token_num = max(1e-3, t_preds_as_label.ne(-100).sum())
        
        t2s_agreement_mask = t2s_logits.argmax(-1).eq(t_preds_as_label)
        t2s_agreement = (t2s_agreement_mask * t_preds_as_label.ne(-100)).sum() / stu_align_token_num
        t2s_agreement_ratio = t2s_agreement_mask.sum() / pad_mask.sum()
        log["t2s_agreement"] = t2s_agreement
        log["t2s_agreement_ratio"] = t2s_agreement_ratio

        t2s_acc_mask = t2s_logits.argmax(-1).eq(target)
        t2s_acc = (t2s_acc_mask * pad_mask).sum() / pad_mask.sum()
        t2s_acc_ratio = t2s_acc_mask.sum() / pad_mask.sum()
        log["t2s_acc"] = t2s_acc
        log["t2s_acc_ratio"] = t2s_acc_ratio
        log["t_preds_as_label_acc"] = (t_preds_as_label.eq(target) * t_preds_as_label.ne(-100)).sum() / stu_align_token_num

        t2s_ce_loss = self.compute_cross_entropy_loss(
            t2s_logits, t_preds_as_label
        )[0] / stu_align_token_num
        t2s_kd_loss = self.dist_func(
            outputs.logits, t2s_logits.detach(), target, reduction="none"
        )
        if t2s_agreement <= distiller.args.t2s_agreement:
            selector = t2s_agreement_mask.float()
        else:
            selector = t_preds_as_label.ne(-100).float()
        weighted_selector = selector * w_s
        denom = weighted_selector.sum().clamp_min(1e-8)
        t2s_kd_loss = (t2s_kd_loss * weighted_selector).sum() / denom
        
        log["t2s_ce_loss"] = t2s_ce_loss
            
        s2t_logits = distiller.teacher_model.lm_head(s2t_hiddens_align)
        s2t_kd_loss = self.dist_func(
            s2t_logits, teacher_outputs.logits, teacher_target, reduction="none"
        )
        valid_teacher_mask = (~s2t_hiddens_align.eq(0).all(-1)).float()
        weighted_teacher_mask = valid_teacher_mask * w_t
        t_denom = weighted_teacher_mask.sum().clamp_min(1e-8)
        s2t_kd_loss = (s2t_kd_loss * weighted_teacher_mask).sum() / t_denom

        if self.args.only_stu_kd:
            kd_loss = t2s_kd_loss + t2s_ce_loss
        elif self.args.only_tea_kd:
            kd_loss = s2t_kd_loss
        else:
            kd_loss = t2s_kd_loss + t2s_ce_loss + s2t_kd_loss
        
        log["t2s_kd_loss"] = t2s_kd_loss
        log["s2t_kd_loss"] = s2t_kd_loss
        log["kd_loss"] = kd_loss

        return kd_loss, log
   

    def compute_on_policy_dual_space_kd_loss_with_eta(
        self, outputs, teacher_outputs, batch, distiller, log
    ):
        target = batch["op_label_batch"]["label"]
        teacher_target = batch["op_teacher_label_batch"]["label"]
          
        pad_mask = target.ne(self.padding_id)
        teacher_pad_mask = teacher_target.ne(self.padding_id)

        hiddens = outputs.hidden_states[-1]
        teacher_hiddens = teacher_outputs.hidden_states[-1]

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
        
        # teacher space
        if self.args.init_s2t_projector:
            stu_lm_head = distiller.student_model.lm_head.weight.detach().transpose(0, 1)
            stu_lm_head = stu_lm_head[:, distiller.student_overlap_token_ids]
            if self.args.topk_vocab != -1:
                stu_lm_head = stu_lm_head[:, :self.args.topk_vocab]
            s2t_projector = stu_lm_head @ distiller.part_teacher_head_pinv
            stu_v_hiddens = hiddens @ s2t_projector
        else:
            stu_v_hiddens = distiller.projectors["s2t"](hiddens)

        tea_v_hiddens = distiller.projectors["t2s"](teacher_hiddens)  # m x D x D x d -> m x d

        t_preds = teacher_outputs.logits.argmax(-1)
        t_preds = torch.where(teacher_pad_mask, t_preds, teacher_target)

        t_preds_as_label = []
        t2s_hiddens_align = torch.zeros_like(tea_v_hiddens).to(tea_v_hiddens.device).to(tea_v_hiddens)   # m x d -> n x d
        s2t_hiddens_align = torch.zeros_like(stu_v_hiddens).to(stu_v_hiddens.device).to(stu_v_hiddens)   # n x D -> m x D
        align_ratio = []
        for i in range(t_preds.shape[0]):
            indices_t_preds = torch.where(t_preds[i] != -100)[0]
            if indices_t_preds.shape[0] == 0:
                t_preds_as_label.append(torch.tensor([-100]*t_preds.shape[-1], device=tea_v_hiddens.device))
                align_ratio.append(1.0)
                continue
            indices_t_target = torch.where(teacher_target[i] != -100)[0]
            indices_s_target = torch.where(target[i] != -100)[0]

            cur_t_preds = t_preds[i][indices_t_preds[0]: indices_t_preds[-1]+1]
            cur_t_target = teacher_target[i][indices_t_target[0]: indices_t_target[-1]+1]
            cur_t_target_tokens = distiller.teacher_tokenizer.convert_ids_to_tokens(cur_t_target)

            cur_s_target = target[i][indices_s_target[0]: indices_s_target[-1]+1]
            cur_s_target_tokens = distiller.student_tokenizer.convert_ids_to_tokens(cur_s_target)

            align_t_idx, align_s_idx = align_sequences(
                cur_t_target_tokens, 
                cur_s_target_tokens,
                distiller.student_tokenizer, 
                distiller.teacher_tokenizer
            )
            if align_t_idx == [] and align_s_idx == []:
                t_preds_as_label.append(torch.tensor([-100]*t_preds.shape[-1], device=tea_v_hiddens.device))
                align_ratio.append(1.0)
                continue
            cur_align_ratio = len(align_s_idx) / len(cur_s_target)
            align_ratio.append(cur_align_ratio)

            cur_t_preds_as_label_1 = target[i][:indices_s_target[0]].cpu().tolist()
            cur_t_preds_as_label_2 = [-100] * len(cur_s_target)
            for _t_idx, _s_idx in zip(align_t_idx, align_s_idx):
                tmp_t_token = distiller.teacher_tokenizer.convert_ids_to_tokens([cur_t_preds[_t_idx]])
                if cur_t_preds[_t_idx] == distiller.teacher_tokenizer.eos_token_id:
                    cur_t_preds_as_label_2[_s_idx] = distiller.student_tokenizer.eos_token_id
                    s2t_hiddens_align[i, _t_idx + indices_t_target[0], :] = stu_v_hiddens[i, _s_idx + indices_s_target[0]]
                    t2s_hiddens_align[i, _s_idx + indices_s_target[0], :] = tea_v_hiddens[i, _t_idx + indices_t_target[0]]
                else:
                    try:
                        tmp = distiller.student_tokenizer.convert_tokens_to_ids(tmp_t_token)
                        if len(tmp) == 1 and tmp[0] is not None:
                            cur_t_preds_as_label_2[_s_idx] = tmp[0]
                            s2t_hiddens_align[i, _t_idx + indices_t_target[0], :] = stu_v_hiddens[i, _s_idx + indices_s_target[0]]
                            t2s_hiddens_align[i, _s_idx + indices_s_target[0], :] = tea_v_hiddens[i, _t_idx + indices_t_target[0]]
                        else:
                            cur_t_preds_as_label_2[_s_idx] = -100
                    except:
                        cur_t_preds_as_label_2[_s_idx] = -100

            assert len(cur_t_preds_as_label_2) == len(cur_s_target)
            cur_t_preds_as_label_3 = target[i][indices_s_target[-1]+1:].cpu().tolist()
            
            cur_t_preds_as_label = cur_t_preds_as_label_1 + cur_t_preds_as_label_2 + cur_t_preds_as_label_3
            cur_t_preds_as_label = cur_t_preds_as_label[:teacher_target.shape[1]]
            t_preds_as_label.append(torch.tensor(cur_t_preds_as_label, device=teacher_target.device))

        t_preds_as_label = torch.cat(t_preds_as_label, dim=0).reshape(-1, teacher_target.shape[1])
        log["align_ratio"] = torch.tensor(align_ratio, device=teacher_target.device).mean()

        t2s_logits = t2s_hiddens_align.matmul(
            distiller.student_model.lm_head.weight.detach().transpose(-1, -2)
        )  # n x d x d x V_stu -> n x V_stu  [bsz x seq-len x V_stu]

        stu_align_token_num = max(1e-3, t_preds_as_label.ne(-100).sum())
        t2s_agreement_mask = t2s_logits.argmax(-1).eq(t_preds_as_label)
        t2s_agreement = (t2s_agreement_mask * t_preds_as_label.ne(-100)).sum() / stu_align_token_num
        t2s_agreement_ratio = t2s_agreement_mask.sum() / pad_mask.sum()
        log["t2s_agreement"] = t2s_agreement
        log["t2s_agreement_ratio"] = t2s_agreement_ratio

        t2s_ce_loss = self.compute_cross_entropy_loss(
            t2s_logits, t_preds_as_label
        )[0] / stu_align_token_num
        t2s_kd_loss = self.dist_func(
            outputs.logits, t2s_logits.detach(), target, reduction="none"
        )
        if t2s_agreement <= distiller.args.t2s_agreement:
            selector = t2s_agreement_mask.float()
        else:
            selector = t_preds_as_label.ne(-100).float()
        weighted_selector = selector * w_s
        denom = weighted_selector.sum().clamp_min(1e-8)
        t2s_kd_loss = (t2s_kd_loss * weighted_selector).sum() / denom

        log["t2s_ce_loss"] = t2s_ce_loss
            
        s2t_logits = distiller.teacher_model.lm_head(s2t_hiddens_align)
        s2t_kd_loss = self.dist_func(
            s2t_logits, teacher_outputs.logits, teacher_target, reduction="none"
        )
        valid_teacher_mask = (~s2t_hiddens_align.eq(0).all(-1)).float()
        weighted_teacher_mask = valid_teacher_mask * w_t
        t_denom = weighted_teacher_mask.sum().clamp_min(1e-8)
        s2t_kd_loss = (s2t_kd_loss * weighted_teacher_mask).sum() / t_denom

        if self.args.only_stu_kd:
            kd_loss = t2s_kd_loss + t2s_ce_loss
        elif self.args.only_tea_kd:
            kd_loss = s2t_kd_loss
        else:
            kd_loss = t2s_kd_loss + t2s_ce_loss + s2t_kd_loss
        
        log["t2s_kd_loss"] = t2s_kd_loss
        log["s2t_kd_loss"] = s2t_kd_loss
        log["kd_loss"] = kd_loss

        return kd_loss, log
   

    def compute_dtw_loss(
        self,
        outputs,
        teacher_outputs,
        distiller,
        student_target,
        teacher_target,
        log,
    ):
        if self.dtw_rate == 0:
            log["dtw_loss"] = 0.0
            return torch.tensor(0.0, device=outputs.logits.device), log

        pad_mask = student_target.ne(self.padding_id)
        teacher_pad_mask = teacher_target.ne(self.padding_id)

        stu_target_embeds, tea_target_embeds = self._get_target_embeddings(
            distiller, student_target, teacher_target, pad_mask, teacher_pad_mask
        )

        hiddens = outputs.hidden_states[-1]
        teacher_hiddens = teacher_outputs.hidden_states[-1]

        projected_teacher_embeds = distiller.projectors["dtw_embed_t2s"](tea_target_embeds)
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

            if (
                self.dtw_band_source == "cma"
                and hasattr(self, "last_align")
                and self.last_align is not None
                and self.dtw_band_width > 0
            ):
                A = self.last_align[i][:s_len, :t_len]
                eps = 1e-9
                A_clamped = (A + eps) / (A.sum(dim=-1, keepdim=True) + eps)
                row_entropy = -(A_clamped * torch.log(A_clamped)).sum(dim=-1)

                lin_center = (
                    torch.arange(s_len, device=A.device, dtype=torch.float32)
                    * (float(t_len) / float(s_len))
                )
                soft_center = (A_clamped * torch.arange(t_len, device=A.device).view(1, -1)).sum(dim=-1)
                alpha = float(self.dtw_band_center_blend)
                centers = alpha * soft_center + (1.0 - alpha) * lin_center

                base_w = float(self.dtw_band_width)
                width = base_w + float(self.dtw_band_entropy_coef) * row_entropy

                j = torch.arange(t_len, device=A.device).view(1, -1).float()
                dist = (j - centers.view(-1, 1)).abs()
                band = dist <= width.view(-1, 1)

                if self.dtw_band_warmup_steps and self.dtw_band_warmup_steps > 0:
                    pen_scale = min(
                        1.0, float(self._global_step + 1) / float(self.dtw_band_warmup_steps)
                    )
                else:
                    pen_scale = 1.0
                penalty = float(self.dtw_band_penalty) * pen_scale
                c_stu_tea = c_stu_tea + (~band).float() * penalty

            if self.dtw_band_source == "sdtw" and self.dtw_band_width > 0:
                _, A = self.dtw.forward_with_cost_matrix(c_stu_tea.unsqueeze(0), return_alignment=True)
                A = A[0]
                eps = 1e-9
                A_clamped = (A + eps) / (A.sum(dim=-1, keepdim=True) + eps)
                row_entropy = -(A_clamped * torch.log(A_clamped + eps)).sum(dim=-1)
                lin_center = (
                    torch.arange(s_len, device=A.device, dtype=torch.float32)
                    * (float(t_len) / float(s_len))
                )
                soft_center = (A_clamped * torch.arange(t_len, device=A.device).view(1, -1)).sum(dim=-1)
                alpha = float(self.dtw_band_center_blend)
                centers = alpha * soft_center + (1.0 - alpha) * lin_center
                base_w = float(self.dtw_band_width)
                width = base_w + float(self.dtw_band_entropy_coef) * row_entropy
                j = torch.arange(t_len, device=A.device).view(1, -1).float()
                dist = (j - centers.view(-1, 1)).abs()
                band = dist <= width.view(-1, 1)
                if self.dtw_band_warmup_steps and self.dtw_band_warmup_steps > 0:
                    pen_scale = min(
                        1.0, float(self._global_step + 1) / float(self.dtw_band_warmup_steps)
                    )
                else:
                    pen_scale = 1.0
                penalty = float(self.dtw_band_penalty) * pen_scale
                c_stu_tea = c_stu_tea + (~band).float() * penalty

            s2t = self.dtw.forward_with_cost_matrix(c_stu_tea.unsqueeze(0))
            s2s = self.dtw.forward_with_cost_matrix(c_stu_stu.unsqueeze(0))
            t2t = self.dtw.forward_with_cost_matrix(c_tea_tea.unsqueeze(0))

            pair_loss = s2t - 0.5 * (s2s + t2t)

            total_loss = total_loss + pair_loss.squeeze()

        if non_empty_pairs == 0:
            return torch.tensor(0.0, device=student_embs.device, requires_grad=True)

        return total_loss

    def _get_target_embeddings(
        self,
        distiller,
        student_target,
        teacher_target,
        pad_mask,
        teacher_pad_mask,
    ):
        if hasattr(distiller.student_model, "model") and hasattr(distiller.student_model.model, "embed_tokens"):
            stu_embed_tokens = distiller.student_model.model.embed_tokens
        elif (
            hasattr(distiller.student_model, "model")
            and hasattr(distiller.student_model.model, "model")
            and hasattr(distiller.student_model.model.model, "embed_tokens")
        ):
            stu_embed_tokens = distiller.student_model.model.model.embed_tokens
        elif hasattr(distiller.student_model, "transformer") and hasattr(distiller.student_model.transformer, "wte"):
            stu_embed_tokens = distiller.student_model.transformer.wte
        else:
            raise NotImplementedError

        if hasattr(distiller.teacher_model, "model") and hasattr(distiller.teacher_model.model, "embed_tokens"):
            tea_embed_tokens = distiller.teacher_model.model.embed_tokens
        elif (
            hasattr(distiller.teacher_model, "model")
            and hasattr(distiller.teacher_model.model, "model")
            and hasattr(distiller.teacher_model.model.model, "embed_tokens")
        ):
            tea_embed_tokens = distiller.teacher_model.model.model.embed_tokens
        elif hasattr(distiller.teacher_model, "transformer") and hasattr(distiller.teacher_model.transformer, "wte"):
            tea_embed_tokens = distiller.teacher_model.transformer.wte
        else:
            raise NotImplementedError

        formal_target = torch.where(pad_mask, student_target, torch.zeros_like(student_target))
        stu_target_embeds = stu_embed_tokens(formal_target)

        formal_teacher_target = torch.where(
            teacher_pad_mask, teacher_target, torch.zeros_like(teacher_target)
        )
        tea_target_embeds = tea_embed_tokens(formal_teacher_target).detach()

        return stu_target_embeds, tea_target_embeds
