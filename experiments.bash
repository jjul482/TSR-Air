python main.py --mode train --gpu 0 --batch_size 16 --max_epochs 100 --horizons 2 --sensor_drop 0 
python main.py --mode train --gpu 1 --batch_size 16 --max_epochs 100 --horizons 4 --sensor_drop 0 
python main.py --mode train --gpu 2 --batch_size 16 --max_epochs 100 --horizons 6 --sensor_drop 0 
python main.py --mode train --gpu 3 --batch_size 16 --max_epochs 100 --horizons 8 --sensor_drop 0
python main.py --mode train --gpu 0 --batch_size 16 --max_epochs 100 --horizons 2 --sensor_drop 10 
python main.py --mode train --gpu 1 --batch_size 16 --max_epochs 100 --horizons 2 --sensor_drop 20 
python main.py --mode train --gpu 2 --batch_size 16 --max_epochs 100 --horizons 2 --sensor_drop 30 
python main.py --mode train --gpu 3 --batch_size 16 --max_epochs 100 --horizons 2 --sensor_drop 40
python main.py --mode train --gpu 0 --batch_size 16 --max_epochs 100 --horizons 2 --sensor_drop 50 
python main.py --mode train --gpu 1 --batch_size 16 --max_epochs 100 --horizons 8 --sensor_drop 0 --data "Delhi" 
python main.py --mode train --gpu 2 --batch_size 16 --max_epochs 100 --horizons 8 --sensor_drop 0 --data "Beijing" 
python main.py --mode train --gpu 3 --batch_size 16 --max_epochs 100 --horizons 8 --sensor_drop 0 --data "Sofia"
python main.py --mode train --gpu 0 --batch_size 16 --max_epochs 100 --horizons 8 --sensor_drop 0 --data "Vinnytsia"

python main.py --mode test --gpu 0 --batch_size 16 --max_epochs 100 --horizons 2 --sensor_drop 0 
python main.py --mode test --gpu 1 --batch_size 16 --max_epochs 100 --horizons 4 --sensor_drop 0 
python main.py --mode test --gpu 2 --batch_size 16 --max_epochs 100 --horizons 6 --sensor_drop 0 
python main.py --mode test --gpu 3 --batch_size 16 --max_epochs 100 --horizons 8 --sensor_drop 0
python main.py --mode test --gpu 0 --batch_size 16 --max_epochs 100 --horizons 2 --sensor_drop 10 
python main.py --mode test --gpu 1 --batch_size 16 --max_epochs 100 --horizons 2 --sensor_drop 20 
python main.py --mode test --gpu 2 --batch_size 16 --max_epochs 100 --horizons 2 --sensor_drop 30 
python main.py --mode test --gpu 3 --batch_size 16 --max_epochs 100 --horizons 2 --sensor_drop 40
python main.py --mode test --gpu 0 --batch_size 16 --max_epochs 100 --horizons 2 --sensor_drop 50 
python main.py --mode test --gpu 1 --batch_size 16 --max_epochs 100 --horizons 8 --sensor_drop 0 --data "Delhi" 
python main.py --mode test --gpu 2 --batch_size 16 --max_epochs 100 --horizons 8 --sensor_drop 0 --data "Beijing" 
python main.py --mode test --gpu 3 --batch_size 16 --max_epochs 100 --horizons 8 --sensor_drop 0 --data "Sofia"
python main.py --mode test --gpu 0 --batch_size 16 --max_epochs 100 --horizons 8 --sensor_drop 0 --data "Vinnytsia"