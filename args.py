#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Command line argument parser for object detection model training.
This module provides a centralized configuration for all training parameters.
"""

import argparse
from typing import Any


def parse_args_and_prepare() -> Any:
    """Parse command line arguments for model training."""
    parser = argparse.ArgumentParser(description="Train and evaluate object detection models")

    # 데이터셋 설정
    parser.add_argument('--data_dir', type=str, default='/Volumes/Macintosh SUB/Dataset/ai03-level1-project', help='데이터셋 디렉토리 경로')
    parser.add_argument('--data_yaml', type=str, default='data.yaml', help='YOLO 학습용 데이터 설정 파일')
    parser.add_argument('--num_classes', type=int, default=73, help='클래스 수 (배경 포함)')

    # 모델 설정
    parser.add_argument('--model_type', type=str, choices=['fasterrcnn', 'yolo11', 'ultralytics'], default='ultralytics', help='사용할 모델 타입')
    parser.add_argument('--yolo_weights', type=str, default='yolo11n.pt', help='YOLO 가중치 파일 경로')
    parser.add_argument('--pretrained', type=bool, default=True, help='사전 학습된 가중치 사용 여부')

    # 학습 하이퍼파라미터
    parser.add_argument('--batch_size', type=int, default=4, help='배치 크기')
    parser.add_argument('--num_epochs', type=int, default=10, help='학습 에폭 수')
    parser.add_argument('--lr', type=float, default=1e-4, help='학습률')
    parser.add_argument('--num_workers', type=int, default=4, help='데이터 로딩 워커 수')

    # 학습 결과 저장 설정
    parser.add_argument('--project', type=str, default='runs/detect', help='결과 저장 프로젝트명')
    parser.add_argument('--name', type=str, default='train', help='실험 이름')
    parser.add_argument('--resume', type=bool, default=False, help='마지막 체크포인트에서 학습 재개')

    return parser.parse_args()
