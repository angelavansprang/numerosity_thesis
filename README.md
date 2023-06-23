# On Numerosity Representations in Vision-Language Models
This repository contains code for the thesis on numerosity in language-vision models. The code is distributed amongst three folders:
- *finetune_clip*. This folder contains code to finetune CLIP to count, as proposed by Paiss et al., 2023, https://teaching-clip-to-count.github.io/.
- *malevic-master*. This folder contains code regarding the MALeViC dataset, which is from Pezzelle & Fernández (2019), https://github.com/sandropezzelle/malevic. The folder also contains code to perform amnesic probing, which is based on Ravfogel et al., 2022, https://github.com/shauli-ravfogel/adv-kernel-removal. Additionally, the folder contains code to train and evaluate probes to reproduce the experiments mentioned in the thesis. Finally, the folder contains some notebooks to visualize counting and bounding boxes of the MALeViC data.
- *style_package*. This folder contains no functional code, but the matplotlib style of the plots in the report.

Note that the files are a combination of re-used and original code. 
