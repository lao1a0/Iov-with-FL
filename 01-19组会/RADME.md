# Soteria

[Soteria/GS_attack at main · jeremy313/Soteria (github.com)](https://github.com/jeremy313/Soteria/tree/main/GS_attack)

##### python reconstruct_image.py --target_id=-1 --defense=ours --pruning_rate=60 --save_image

```bash
Currently evaluating -------------------------------:
Friday, 12. January 2024 09:03PM
CPUs: 18, GPUs: 1 on tsz-server-TITIAN.
Namespace(model='ConvNet', dataset='CIFAR10', dtype='float', trained_model=False, epochs=120, accumulation=0, num_images=1, target_id=-1, label_flip=False, restarts=1, cost_fn='sim', indices='def', weights='equal', optimizer='adam', signed=True, boxed=True, scoring_choice='loss', init='randn', tv=0.0001, save_image=True, image_path='images_new/', model_path='models/', data_path='~/data', name='iv', deterministic=False, dryrun=False, defense='ours', pruning_rate=60.0)
GPU : TITAN RTX
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to /home/raoxy/data/cifar-10-python.tar.gz
170500096it [00:14, 11519798.01it/s]                                                                                                                                                        
Extracting /home/raoxy/data/cifar-10-python.tar.gz to /home/raoxy/data
Files already downloaded and verified
Model initialized with random key 3045196091.
/home/raoxy/Soteria/GS_attack/reconstruct_image.py:64: DeprecationWarning: BICUBIC is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.BICUBIC instead.
  ground_truth = torch.as_tensor(np.array(Image.open("auto.jpg").resize((32, 32), Image.BICUBIC)) / 255, **setup)
torch.Size([1, 3, 32, 32])
applying our defense strategy...
Full gradient norm is 4.521341e-02.
It: 0. Rec. loss: 0.0341.
It: 500. Rec. loss: 0.0293.
It: 1000. Rec. loss: 0.0286.
It: 1500. Rec. loss: 0.0277.
.......
It: 23000. Rec. loss: 0.0284.
It: 23500. Rec. loss: 0.0285.
It: 23999. Rec. loss: 0.0292.
Choosing optimal result ...
Optimal result score: 0.0288
Total time: 428.6142177581787.
Rec. loss: 0.0288 | MSE: 1.0726 | PSNR: 11.72 | FMSE: 5.0144e-08 |
Friday, 12. January 2024 09:11PM
---------------------------------------------------
Finished computations with time: 0:07:33.157648
-------------Job finished.-------------------------
```

##### python reconstruct_image.py --target_id=-1 --defense=prune --pruning_rate=60 --save_image

```bash
Currently evaluating -------------------------------:
Friday, 12. January 2024 09:24PM
CPUs: 18, GPUs: 1 on tsz-server-TITIAN.
Namespace(model='ConvNet', dataset='CIFAR10', dtype='float', trained_model=False, epochs=120, accumulation=0, num_images=1, target_id=-1, label_flip=False, restarts=1, cost_fn='sim', indices='def', weights='equal', optimizer='adam', signed=True, boxed=True, scoring_choice='loss', init='randn', tv=0.0001, save_image=True, image_path='images_new/', model_path='models/', data_path='~/data', name='iv', deterministic=False, dryrun=False, defense='prune', pruning_rate=60.0)
GPU : TITAN RTX
Files already downloaded and verified
Files already downloaded and verified
Model initialized with random key 713168903.
/home/raoxy/Soteria/GS_attack/reconstruct_image.py:64: DeprecationWarning: BICUBIC is deprecated and will be removed in Pillow 10 (2023-07-01). Use Resampling.BICUBIC instead.
  ground_truth = torch.as_tensor(np.array(Image.open("auto.jpg").resize((32, 32), Image.BICUBIC)) / 255, **setup)
torch.Size([1, 3, 32, 32])
Full gradient norm is 4.499292e-02.
It: 0. Rec. loss: 0.0349.
It: 500. Rec. loss: 0.0271.
............
It: 23999. Rec. loss: 0.0286.
Choosing optimal result ...
Optimal result score: 0.0285
Total time: 413.68210530281067.
Rec. loss: 0.0285 | MSE: 0.2122 | PSNR: 18.75 | FMSE: 1.0052e-07 |
Friday, 12. January 2024 09:31PM
---------------------------------------------------
Finished computations with time: 0:06:56.982207
-------------Job finished.-------------------------
```



# ATSPrivacy



```
pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install transformers==4.22.2 datasets==2.5.2
```

