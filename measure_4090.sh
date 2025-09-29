#!/usr/bin/env bash
python -m wnet2d.utils.measure --model wnet2d --size 512 --precision fp32 --runs 200 --warmup 50 --report reserved
