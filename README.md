# Multi-Channel Target Speech Extraction with Channel Decorrelation and Target Speaker Adaptation

The codes here are implementations of two methods for exploiting the multi-channel spatial information to extract the target speech. 
The first one is using a target speech adaptation layer in a parallel encoder architecture. The second one is designing a channel decorrelation mechanism to extract the inter-channel differential information to enhance the multi-channel encoder representation.

# Data 

- Mixtures, `cd ./data/anechoic`, modify  corresponding data path and run `sh run_data_generation.sh`.
- Reverberation, `cd ./data/reverberate`, modify  corresponding data path and run `sh launch_spatialize.sh`.

    
# Usage

- training: configure 
- evaluate & separate

# Results

| System | IPD  | Adap | SDR/SiSDR |
| :------| :--: | :--: | :-------: |  
|(0) TSB | - | - | 11.51 / 11.00 |
|(1)   |   Y   | -   | 11.57 / 11.07 |
|(2) Parallel  |   -   | -   | 12.43 / 11.91 |
|(3)  |  -    | Y   | 12.73 / 12.20 |
|(4) CD  |  -    | -   | 12.87 / 12.34 |
|(5)   |     - | Y   | 12.87 / 12.35 |
|(6)   |   Y   | Y   | 12.55 / 12.01 |
|(7) CC  |    -  | Y   | 12.66 / 12.13 |


# Contact
Email: jyhan03@163.com

# Reference
- Code & Scripts
    - [Mixture data](https://github.com/xuchenglin28/speaker_extraction)
    - [Reverberate data](https://www.merl.com/demos/deep-clustering)
    - [Conv-TasNet implementation](https://github.com/funcwj/conv-tasnet)
- Paper
    -  [M. Delcroix, T. Ochiai, K. Zmolikova, K. Kinoshita, N. Tawara, T.Nakatani, and S. Araki, “Improving speaker discrimination of target speech extraction with time-domain speakerbeam,” in Proc. ICASSP. IEEE, 2020, pp. 691–695.](https://arxiv.org/abs/2001.08378)







 
