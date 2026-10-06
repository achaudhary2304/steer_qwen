"""Signed additive next-token contributions, including numerical rounding."""
import torch


@torch.no_grad()
def token_attribution(model, module, parts, reconstructed, token_id, residual_scale,
                      corpus, native_logit, final_logit, interventions=None):
    w = model.head.weight[token_id].float()
    activations = parts.known_activations[0].clone()
    for index,value in (interventions or {}).items():
        activations[index] = value
    known = activations * (module.known_vectors.float() @ w)
    unknown = parts.unknown_activations[0] * (module.unknown_left.float() @ (module.unknown_right.float() @ w))
    unknown_bias = float(module.unknown_bias.float() @ w)
    residual = float(residual_scale * (parts.residual[0].float() @ w))
    head_bias = float(model.head.bias[token_id]) if getattr(model.head,'bias',None) is not None else 0.
    adjustment = final_logit-native_logit
    named_sum, unknown_sum = float(known.sum()), float(unknown.sum())+unknown_bias
    reconstructed_sum = named_sum+unknown_sum+residual+head_bias+adjustment
    def strongest(values):
        indices = values.abs().topk(min(5,len(values))).indices.tolist()
        return [{'index':i,'contribution':float(values[i])} for i in indices if values[i] != 0]
    named = strongest(known)
    for item in named:
        i = item['index']
        item.update(atlas_id=corpus.ids[i],name=corpus.manifest['concepts'][i]['name'])
    return {'token_id':token_id,'actual_logit':final_logit,
        'components':{'named':named_sum,'unknown':unknown_sum,'residual':residual,
                      'head_bias':head_bias,'steering_logit_adjustment':adjustment},
        'rounding_gap':final_logit-reconstructed_sum,
        'largest_named_contributions':named,'largest_unknown_contributions':strongest(unknown),
        'unknown_bias_contribution':unknown_bias,
        'caveat':'Additive output contributions, not a complete reasoning explanation; chunk-trained names may not describe each token.'}
