# DAVEnet Pytorch

Implementation in Pytorch of the DAVEnet (Deep Audio-Visual Embedding network) model, as described in

David Harwath, Adrià Recasens, Dídac Surís, Galen Chuang, Antonio Torralba, and James Glass, "Jointly Discovering Visual Objects and Spoken Words from Raw Sensory Input," ECCV 2018

## Run the code
```
python run_phone.py --losstype choose_loss_type --feature choose_feature_type --data_dir Your_data_path
```
* losstype: you can choose from ['triplet', 'mml','DAMSM','tripop'], where tripop is also triplet loss
* feature: you can choose from ['tensor','vector'], where 'tensor' uses the architecture of DAVEnet, and 'vector' uses Danny's architecture

For the details of parameters, pleas check *.sh files in ./run

