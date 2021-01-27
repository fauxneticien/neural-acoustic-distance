import torch
import fairseq
import soundfile as sf
import os

local_repo_path = "/Users/fauxneticien/git-repos/neural-acoustic-distance/"
layer_i = 10
w2v2_variant = "base" 
wav, sr = sf.read(os.path.join(local_repo_path, 'scripts', 'test_file.wav'))

wav = torch.from_numpy(wav).unsqueeze(0).float()

# pt files downloaded from https://github.com/pytorch/fairseq/tree/master/examples/wav2vec#pre-trained-models
cp_filenames = {
    'base': 'xlsr_53_56k.pt',      # Wav2Vec 2.0 Base, No finetuning
    'large': 'wav2vec_vox_new.pt', # Wav2Vec 2.0 Large (LV-60), No finetuning
    'xlsr': 'xlsr_53_56k.pt'       # Multilingual pre-trained wav2vec 2.0 (XLSR) models
}

cp_path = os.path.join(local_repo_path, 'scripts', cp_filenames[w2v2_variant])

# Load model according to latest fairseq recommendation: 
# https://github.com/pytorch/fairseq/tree/master/examples/wav2vec#example-usage-1
model, cfg = fairseq.checkpoint_utils.load_model_ensemble([cp_path])
model = model[0]
model.eval()

# # stuff from Martijn's extract_wav2vec2_hiddens function
x = model.feature_extractor(wav)

x = x.transpose(1, 2)
x = model.layer_norm(x)
x = model.post_extract_proj(x)

x_conv = model.encoder.pos_conv(x.transpose(1, 2))
x_conv = x_conv.transpose(1, 2)
x += x_conv

x = model.encoder.layer_norm(x)
x = x.transpose(0, 1)

print("From %i layers, selecting layer %i" % (len(model.encoder.layers), layer_i))

for i, layer in enumerate(model.encoder.layers):
    x, z = layer(x, self_attn_padding_mask=None, need_weights=False)
    if i == layer_i:
        break

x = x.transpose(0, 1)

# stuff from Martijn's extract_features function

# layer_i only seems to be relevant to the "quantize and aggregate and self.method == 'w2v2-qa'" condition:
#
# elif quantize and aggregate and self.method == 'w2v2-qa':
#    z = extract_wav2vec2_hiddens(self.model, wav, layer_i)
#    z = torch.transpose(z, 1, 2)
#
# so I'm transposing the result of x (output of extract_wav2vec2_hiddens) in the same way

x = torch.transpose(x, 1, 2)

print(x.shape)
# shape of x is (1, 1024, 43) representing 1 feature matrix with 1024 features for 43 time steps
# for a wav file (test_file.wav) 
