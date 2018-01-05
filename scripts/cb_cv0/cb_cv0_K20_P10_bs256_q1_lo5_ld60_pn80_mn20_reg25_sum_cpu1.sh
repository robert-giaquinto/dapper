#!/bin/bash
python -m src.experiments.cv0 --num_topics 20 --num_personas 10 \
    --regularization 0.25 \
    --normalization sum \
    --batch_size 256 \
    --queue_size 1 \
    --em_max_iter 50 \
    --max_training_minutes 1440.0 \
    --measurement_noise 0.8 --process_noise 0.2 \
    --learning_offset 5 --learning_decay 0.6 ;
