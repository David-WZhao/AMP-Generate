This is the codes of the AMP-Generate in our manuscript "A Novel Framework for Generating Pathogen-targeted Antimicrobial Peptides with Programmable Physicochemical Properties".

## Requirement

1. python==3.8.15
1. pytorch==1.12.1
1. torchvision==0.12.1
1. torchaudio==0.12.1
1. transformer==4.31.0
1. biopython == 1.79
1. tqdm == 4.66.5

## Train Condition VAE

The training set data exceeds 100MB and cannot be uploaded. If you need any information, please contact us by email.

```
torchrun main.py --work TransVAE 
```

## Train Diffusion Model

```
python main.py --work GetMem_nc
python combine_mem.py
torchrun main.py --work LatentDiffusion_nocondition 
```

## Fine-tune Diffusion(Target E. coli)

```
python main_Ecoli.py --work GetMem_c
python combine_mem_c.py
torchrun main_Ecoli.py --work LatentDiffusion_condition 
```

## Fine-tune Diffusion(Target S. aureus)

``` 
python main_Saureus.py --work GetMem_c
python combine_mem_c.py
torchrun main_Saureus.py --work LatentDiffusion_condition 
```

## Generation(Target E. coli or S. aureus)

``` 
torchrun main_Ecoli.py --work Generate
torchrun main_Saureus.py --work Generate
```

## MIC Predictor

The input requirement is a fasta file.

```
cd Filter/
cd Ecoli or Saureus
python predict.py 
```
