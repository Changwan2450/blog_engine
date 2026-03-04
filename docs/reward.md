# Reward Function

이 문서는 학습 기준이다.

## Metrics

- views
- likes
- comments

## Reward Formula

`reward = (comments * 2) + (likes * 1) + (views * 0.01)`

## Example

- views = 1200
- likes = 34
- comments = 5

`reward = (5 * 2) + (34 * 1) + (1200 * 0.01)`

`reward = 10 + 34 + 12`

`reward = 56`

## Learning Cycle

1. Record metrics
2. Calculate reward
3. Update bandit weights
