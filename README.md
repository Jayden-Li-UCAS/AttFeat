# AttFeat
-AttFeat: Attention-based Features for Infrared and Visible Remote Sensing Image Matching

-Infrared and visible remote sensing image matching is significant for utilizing remote sensing images to obtain scene information. However, due to the large number of sparse and repetitive texture regions in multi-modal remote sensing scenes, feature extraction poses serious difficulties. To address these challenges, this letter proposed the Attention-based 
 Features (AttFeat). First, to solve the problem of coarse feature representation, we proposed the Parallel Channel and Spatial Attention (PCSA) Module, which focuses on important spatial locations and provides richer cross-channel representation. Second, to address the lack of contextual information, we proposed the Squeezed-Axis Transformer (SAFormer), which 
 can obtain a dense global receptive field at a lower cost while retaining rich details. Finally, we use a multi-modal recoupling loss to optimize the relationship between the rich feature description and large receptive field. Extensive experiments on aviation and remote sensing multi-modal datasets demonstrate the superiority of our algorithm and the 
 effectiveness of the proposed modules. In terms of detection and matching performance, AttFeat outperforms the baseline ReDFeat by 10.81% and 15.48% respectively.

-Our weights are obtained from training on the following two datasets.
 The Utah AGRC dataset：https://downloads.greyc.fr/vedai/
 The Amazon Rainforest Satellite Monitoring dataset：https://www.kaggle.com/competitions/planet-understanding-the-amazon-from-space/data

-Thanks to the contributions of the RedFeat algorithm, which can be referenced and cited in detail in "ReDFeat: Recoupling Detection and Description for Multimodal Feature Learning."


