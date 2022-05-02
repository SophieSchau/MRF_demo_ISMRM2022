import numpy as np
import collections
import torch
import networks

import params

def synthesize_star(input, idx, weights):
  '''
  synth = synthesize_star(input, idx, weights)

  Synthesize clinical contrasts using startGAN synthesis network.

  Inputs:
    rec (Numpy array): 

  Returns:
    synth (Array): Processed volume. Shape [256 256 256 size(idx)]
  '''
  netG = networks.define_G([], np.shape(input)[0], params.output_nc,
                           params.ngf, params.which_model_netG,
                           params.norm, params.dropout, params.init_type,
                           params.init_gain, params.gpu_ids)
  state_dict = torch.load("/mnt/" + weights)

  new_state_dict = collections.OrderedDict() 
  for k, v in state_dict.items(): 
    name = k.replace('model', 'module.model')
    new_state_dict[name] = v
  netG.load_state_dict(new_state_dict)
  #netG.eval() # TODO: This made the network not work?

  #x = np.concatenate((input, \
    #np.zeros((1, input.shape[1], input.shape[2], input.shape[3]))), axis=0)
  x = input
  x = np.transpose(x, (3, 0, 1, 2))
  x = torch.from_numpy(x).to(params.gpu_ids[0], dtype=torch.float)

  x = torch.split(x, 1) # TODO: Using larger batch sizes causes errors when evaluating netG.

  out = []
  for k in idx:
    label_channel = torch.zeros((1, 6)).to(params.gpu_ids[0],
                                           dtype=torch.float)
    label_channel[0][k] = 1

    res = np.zeros(input.shape[1:])
    for z in range(res.shape[0]):
      res[z, ...] = netG(x[z], label_channel).detach().cpu().squeeze()
    out.append(res)

  return np.array(out)
