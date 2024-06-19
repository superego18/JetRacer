import matplotlib.pyplot as plt

# 데이터 파일 경로
file_path = 'blue_red_points.txt'

# 빈 리스트
blue_points = []
red_points = []

# 파일 읽기
with open(file_path, 'r') as file:
    for line in file:
        line = line.strip()
        if line.startswith('Blue'):
            _, blue_data, _, red_data = line.split(': ')
            blue_points.append(eval(blue_data))
            red_points.append(eval(red_data))

# 데이터 추출 확인
print("Blue Points:", blue_points)
print("Red Points:", red_points)

# 플롯 설정
plt.figure(figsize=(8, 6))

# Blue와 Red Points 플롯
for blue_point, red_point in zip(blue_points, red_points):
    plt.scatter(blue_point[0], blue_point[1], color='blue', label='Blue Points')
    plt.scatter(red_point[0], red_point[1], color='red', label='Red Points')

# 축 레이블 및 타이틀 설정
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Blue and Red Points')

# 범례 추가
plt.legend()

# 플롯 보이기
plt.grid(True)
plt.show()
