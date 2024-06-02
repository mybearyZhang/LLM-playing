import torch
from LongformerGPT2 import LongformerGPT2LMHeadModel, LongformerGPT2Config

param_list = ['h.0.attn.query.weight', 'h.0.attn.query.bias', 'h.0.attn.key.weight', 'h.0.attn.key.bias', 'h.0.attn.value.weight',
              'h.0.attn.value.bias', 'h.0.attn.query_global.weight', 'h.0.attn.query_global.bias', 'h.0.attn.key_global.weight',
              'h.0.attn.key_global.bias', 'h.0.attn.value_global.weight', 'h.0.attn.value_global.bias', 'h.1.attn.query.weight',
              'h.1.attn.query.bias', 'h.1.attn.key.weight', 'h.1.attn.key.bias', 'h.1.attn.value.weight', 'h.1.attn.value.bias',
              'h.1.attn.query_global.weight', 'h.1.attn.query_global.bias', 'h.1.attn.key_global.weight', 'h.1.attn.key_global.bias',
              'h.1.attn.value_global.weight', 'h.1.attn.value_global.bias', 'h.2.attn.query.weight', 'h.2.attn.query.bias',
              'h.2.attn.key.weight', 'h.2.attn.key.bias', 'h.2.attn.value.weight', 'h.2.attn.value.bias', 'h.2.attn.query_global.weight',
              'h.2.attn.query_global.bias', 'h.2.attn.key_global.weight', 'h.2.attn.key_global.bias', 'h.2.attn.value_global.weight',
              'h.2.attn.value_global.bias', 'h.3.attn.query.weight', 'h.3.attn.query.bias', 'h.3.attn.key.weight', 'h.3.attn.key.bias',
              'h.3.attn.value.weight', 'h.3.attn.value.bias', 'h.3.attn.query_global.weight', 'h.3.attn.query_global.bias',
              'h.3.attn.key_global.weight', 'h.3.attn.key_global.bias', 'h.3.attn.value_global.weight', 'h.3.attn.value_global.bias',
              'h.4.attn.query.weight', 'h.4.attn.query.bias', 'h.4.attn.key.weight', 'h.4.attn.key.bias', 'h.4.attn.value.weight',
              'h.4.attn.value.bias', 'h.4.attn.query_global.weight', 'h.4.attn.query_global.bias', 'h.4.attn.key_global.weight',
              'h.4.attn.key_global.bias', 'h.4.attn.value_global.weight', 'h.4.attn.value_global.bias', 'h.5.attn.query.weight',
              'h.5.attn.query.bias', 'h.5.attn.key.weight', 'h.5.attn.key.bias', 'h.5.attn.value.weight', 'h.5.attn.value.bias',
              'h.5.attn.query_global.weight', 'h.5.attn.query_global.bias', 'h.5.attn.key_global.weight', 'h.5.attn.key_global.bias',
              'h.5.attn.value_global.weight', 'h.5.attn.value_global.bias', 'h.6.attn.query.weight', 'h.6.attn.query.bias',
              'h.6.attn.key.weight', 'h.6.attn.key.bias', 'h.6.attn.value.weight', 'h.6.attn.value.bias', 'h.6.attn.query_global.weight',
              'h.6.attn.query_global.bias', 'h.6.attn.key_global.weight', 'h.6.attn.key_global.bias', 'h.6.attn.value_global.weight',
              'h.6.attn.value_global.bias', 'h.7.attn.query.weight', 'h.7.attn.query.bias', 'h.7.attn.key.weight', 'h.7.attn.key.bias',
              'h.7.attn.value.weight', 'h.7.attn.value.bias', 'h.7.attn.query_global.weight', 'h.7.attn.query_global.bias',
              'h.7.attn.key_global.weight', 'h.7.attn.key_global.bias', 'h.7.attn.value_global.weight', 'h.7.attn.value_global.bias',
              'h.8.attn.query.weight', 'h.8.attn.query.bias', 'h.8.attn.key.weight', 'h.8.attn.key.bias', 'h.8.attn.value.weight',
              'h.8.attn.value.bias', 'h.8.attn.query_global.weight', 'h.8.attn.query_global.bias', 'h.8.attn.key_global.weight',
              'h.8.attn.key_global.bias', 'h.8.attn.value_global.weight', 'h.8.attn.value_global.bias', 'h.9.attn.query.weight',
              'h.9.attn.query.bias', 'h.9.attn.key.weight', 'h.9.attn.key.bias', 'h.9.attn.value.weight', 'h.9.attn.value.bias',
              'h.9.attn.query_global.weight', 'h.9.attn.query_global.bias', 'h.9.attn.key_global.weight', 'h.9.attn.key_global.bias',
              'h.9.attn.value_global.weight', 'h.9.attn.value_global.bias', 'h.10.attn.query.weight', 'h.10.attn.query.bias',
              'h.10.attn.key.weight', 'h.10.attn.key.bias', 'h.10.attn.value.weight', 'h.10.attn.value.bias', 'h.10.attn.query_global.weight',
              'h.10.attn.query_global.bias', 'h.10.attn.key_global.weight', 'h.10.attn.key_global.bias', 'h.10.attn.value_global.weight',
              'h.10.attn.value_global.bias', 'h.11.attn.query.weight', 'h.11.attn.query.bias', 'h.11.attn.key.weight', 'h.11.attn.key.bias',
              'h.11.attn.value.weight', 'h.11.attn.value.bias', 'h.11.attn.query_global.weight', 'h.11.attn.query_global.bias',
              'h.11.attn.key_global.weight', 'h.11.attn.key_global.bias', 'h.11.attn.value_global.weight', 'h.11.attn.value_global.bias']

longformer_state_dict = torch.load('./longformer-base-4096/pytorch_model.bin')

config = LongformerGPT2Config.from_pretrained('./gpt2_model/')
model = LongformerGPT2LMHeadModel.from_pretrained('./gpt2_model/', config=config)

state_dict = model.state_dict()
# transformer.h.1.x.key.weight -> roberta.encoder.layer.1.attention.self.key.weight
for param in param_list:
    param_partition = param.split('.')
    longformer_param = '.'.join(['roberta', 'encoder', 'layer', param_partition[1], 'attention', 'self', param_partition[3], param_partition[4]])
    if 'transformer.' + param in state_dict and longformer_param in longformer_state_dict:
        state_dict['transformer.' + param] = longformer_state_dict[longformer_param]
    else:
        print(f'ERROR: Parameter {param} not found !!!')

torch.save(state_dict, './longformer_gpt2_model/pytorch_model.bin')
