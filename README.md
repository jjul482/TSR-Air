# TSR-Air

## Abstract
A decrease in air quality presents a significant hazard to the sustainability of environmental conditions in the modern world. Its significance in shaping health outcomes and the quality of life in urban areas is projected to escalate over time. Many factors, encompassing anthropogenic emissions and natural phenomena, are recognised as primary influencers contributing to the escalation of air pollution levels. Human health is particularly threatened by high amounts of pollution caused by weather events or disasters. However, these extreme events are difficult to predict using machine learning techniques due to their rapid onset and rarity. Our paper aims to provide an analysis and predictive model of extreme air quality events. By identifying tasks within air quality environments, we deploy continual learning approaches such as replay to prevent forgetting of identified tasks. We present a replay-based forecasting method, Time Sensitive Replay (TSR), to capture and reintroduce dangerous patterns of air quality readings to improve model recollection of those extreme events. We also show that our method is comparable to existing baselines while forecasting non-extreme data points.

## Framework
![alt text](framework.png?raw=true)

## Requirements
- Pytorch 1.10
- Numpy and Scipy
- Pandas
- CUDA 11.3 w/ cuDNN

## Experiments
To run the code, execute the following command:
  bash experiments.bash
