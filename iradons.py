# 필수라이브러리
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astra
import random
import cv2
from skimage.metrics import structural_similarity as ssim
import iradons



'''lib version anaconda 기반

Pandas Version      : 1.4.4
numpy Version       : 1.24.2
matplotlib Version  : 3.5.3
cv2 Version         : 4.7.0
astra Version       : 2.1.0
scikit-image Version: 0.19.2

'''


def phontom_mask(recon, r=150):
    """r = 반지름
       recon = reconstruction된 image
       (x,y) = 원의 중심 좌표
     """
    recon3 = pd.DataFrame(recon)
    # 원 만들기
    for x in range(0, r * 2):
        # 원의 방정식
        y = np.sqrt(r ** 2 - (x - 149) ** 2) + 149
        # 원 최외각부 오른쪽 반원 좌표
        a1 = 149 - int((y - 149).round())
        # 원 최외각부 왼쪽 반원좌표
        a2 = int(y.round(0)) + 1
        # 원에 해당하지 않는 부분 value 0으로 변경
        recon3.loc[(x), 0:a1] = 0
        recon3.loc[(x), a2:300] = 0

    recon3 = np.array(recon3)

    return recon3


def genGaussianKernel(width, sigma):
    array = np.arange((width // 2) * (-1), (width // 2) + 1)
    # 중심에서부터의 거리 계산
    arr = np.zeros((width, width))
    # x^2+y^2 부분을 미리 계산해둘 매트릭스 initialize
    for x in range(width):
        for y in range(width):
            arr[x, y] = array[x] ** 2 + array[y] ** 2
            # 중심에서부터의 거리를 제곱합으로 계산
    kernel_2d = np.zeros((width, width))
    # 커널의 값을 저장할 매트릭스 생성
    for x in range(width):
        for y in range(width):
            kernel_2d[x, y] = np.exp(-arr[x, y] / (2 * sigma ** 2))
            # 수식에 맞게 값 저장(역시나 상수 부분은 생략)
    kernel_2d /= kernel_2d.sum()
    # 전체 값의 합으로 나누어 필터 전체의 합이 1이 되도록 함
    return kernel_2d

def gaus_block(size = 60, sigma = 10, maxvalue = 1.8, step =0.1):
    '''size : 생성될 filer block 크기
    sigma : Gaussian 분포에서 sigma 값 (분포의 형태)
    maxvalue : Gaussian에서 최대값
    step : 다른 Gaussian filter간 떨어진 정도
    ex) step = 0.2 이면 각 filter의 최대값은 1.8, 1.6, 1.4, 1.2순으로 형성
    '''
    # 입력된 최대값과 출력된 최대값 맞추기
    maxvalue = maxvalue - 1

    # gaus 블록 정의
    gaus1 = genGaussianKernel(size, sigma)  # Gaussian 최대값 1.8 기본값
    gaus1 = (gaus1 / gaus1.max() + 1) * maxvalue + (1 - maxvalue)  # # Gaussian 범위 보정 1 ~ 최대값까지

    gaus2 = genGaussianKernel(size, sigma)  # Gaussian 최대값 1.7 기본값
    gaus2 = (gaus2 / gaus2.max() + 1) * (maxvalue - step) + (1 - maxvalue + step)

    gaus3 = genGaussianKernel(size, sigma)  # Gaussian 최대값 1.6 기본값
    gaus3 = (gaus3 / gaus3.max() + 1) * (maxvalue - step * 2) + (1 - maxvalue + step * 2)

    gaus4 = genGaussianKernel(size, sigma)  # Gaussian 최대값 1.5 기본값
    gaus4 = (gaus4 / gaus4.max() + 1) * (maxvalue - step * 3) + (1 - maxvalue + step * 3)

    nogaus = np.ones((size, size))
    gaus_list = [nogaus, gaus1, gaus2, gaus3, gaus4]
    return gaus_list





# filtered phantom 생성
def Gaussianfilter(gaus_list, phantom, train_lastnum=1, percent=16):
    '''
    gaus_list : gaus_block list
    phantom : 필터에 사용할 phantom
    train_lastnum : 만들 data 수
    percent : filter 생성확률
    '''
    # filtered phantom 생성 시작
    i = 0
    # 확률 조정
    percentage = int(4 * 100 / percent) + 1

    for i in range(train_lastnum):
        for num2 in [1, 2, 3, 4, 5]:
            for num in [1, 2, 3, 4, 5]:

                # randint 함수로 확률에 사용될 난수 생성
                x = random.randint(1, percentage)
                # 난수에 따라 gaus 분포 결정
                if x == 1:
                    y = gaus_list[4]
                if x == 2:
                    y = gaus_list[3]
                if x == 3:
                    y = gaus_list[2]
                if x == 4:
                    y = gaus_list[1]
                if x >= 5:
                    y = gaus_list[0]  # Gaussianfilter 생성 생략 변수

                # 필터 생성위치 정의
                if num == 1:
                    a = y
                if num == 2:
                    b = y
                if num == 3:
                    c = y
                if num == 4:
                    d = y
                if num == 5:
                    e = y

            # 왼쪽부터 오른쪽으로 필터 5개 생성(변수당 1행 5열)
            if num2 == 1:
                gau_st1 = np.hstack((a, b, c, d, e))
            if num2 == 2:
                gau_st2 = np.hstack((a, b, c, d, e))
            if num2 == 3:
                gau_st3 = np.hstack((a, b, c, d, e))
            if num2 == 4:
                gau_st4 = np.hstack((a, b, c, d, e))
            if num2 == 5:
                gau_st5 = np.hstack((a, b, c, d, e))

        # 1행 5열 필터를 5행 5열로 결합
        test_gau_final = np.vstack((gau_st1, gau_st2, gau_st3, gau_st4, gau_st5))
        # 필터와 phantom 결합
        phantom_gaus = phantom * test_gau_final
        # train data 저장, 경로는 기본값(dir 함수 사용할 때 나오는 위치)
        np.save(f'filter_{i}.npy', phantom_gaus)
    i += 1
    return phantom_gaus


# 타원 가우스 필터
def Gaussianfilter_ep(gaus_list, phantom, train_lastnum=1, percent=16):
    '''
    gaus_list : gaus_block list
    phantom : 필터에 사용할 phantom
    train_lastnum : 만들 data 수
    percent : filter 생성확률
    '''
    # filtered phantom 생성 시작
    i = 0
    [x_size, y_size] = phantom.shape
    # 확률 조정
    percentage = int(4 * 100 / percent) + 1

    for i in range(train_lastnum):
        for num2 in [1, 2, 3, 4, 5, 6, 7]:
            for num in [1, 2, 3, 4]:

                # randint 함수로 확률에 사용될 난수 생성
                x = random.randint(1, percentage)
                # 난수에 따라 gaus 분포 결정
                if x == 1:
                    y = gaus_list[4]
                if x == 2:
                    y = gaus_list[3]
                if x == 3:
                    y = gaus_list[2]
                if x == 4:
                    y = gaus_list[1]
                if x >= 5:
                    y = gaus_list[0]  # Gaussianfilter 생성 생략 변수

                # 필터 생성위치 정의
                if num == 1:
                    a = y
                if num == 2:
                    b = y
                if num == 3:
                    c = y
                if num == 4:
                    d = y

            # 왼쪽부터 오른쪽으로 필터 5개 생성(변수당 1행 5열)
            if num2 == 1:
                gau_st1 = np.hstack((a, b, c, d))
            if num2 == 2:
                gau_st2 = np.hstack((a, b, c, d))
            if num2 == 3:
                gau_st3 = np.hstack((a, b, c, d))
            if num2 == 4:
                gau_st4 = np.hstack((a, b, c, d))
            if num2 == 5:
                gau_st5 = np.hstack((a, b, c, d))
            if num2 == 6:
                gau_st6 = np.hstack((a, b, c, d))
            if num2 == 7:
                gau_st7 = np.hstack((a, b, c, d))

        # 1행 5열 필터를 5행 5열로 결합
        test_gau_final = np.vstack((gau_st1, gau_st2, gau_st3, gau_st4, gau_st5, gau_st6, gau_st7))
        # 필터와 phantom 결합
        phantom_gaus = phantom * test_gau_final[0:x_size, 0:y_size]
        # train data 저장, 경로는 기본값(dir 함수 사용할 때 나오는 위치)
        np.save(f'filter_{i}.npy', phantom_gaus)
    i += 1
    return phantom_gaus


def make_sino(phantom, r=150, det_count=7, use_source='no', source=3, fixed_angles=[0, 70, 290], theta_degree=30
              , beam_type='fanflat', kernal_type='strip_fanflat'):
    '''

    phantom : sinogram 만들 phantom
    r : 원의 반지름
    반드시 phantom과 동일한 반지름이어야 함

    det_count : detector 수
    use_source : 광원에 동일한 각도 주는 경우
    ex) source = 3이면 angle = [0,120,240] 각도 적용
    source : 광원의 수
    fixed_angles : use_source = 'no'로 지정한 경우 angle 직접 작성

    theta : 원과 두 접선이 만나는 각도의 1/2
    값은 1 ~ 85 사이값 추천

    beam_type : 광원의 beam type
    지원 beam type : fanflat, parallel
    kernal_type = 광원 생성 방법
    지원 kanel : strip_fanflat, line_fanflat
    '''

    # config data
    # recontruction phantom size 정의
    phantom_use = phantom / phantom.max() * 255
    [x_size, y_size] = phantom_use.shape
    # 각로 radian 변환
    theta = np.radians(theta_degree)

    # source에서 orign까지 거리
    source_orign = r / np.sin(theta)
    # origin에서 detector까지 거리
    origin_det = r

    # create geometries and projector
    if beam_type == 'fanflat':
        # source이용(fixed_angles 정의 불필요)
        if use_source == 'yes':
            angles = np.linspace(0, 360, source, False) * (np.pi / 180)  # radian으로 변환

        # angle 고정용(fixed_angles 변수를 정의 해야함)
        if use_source == 'no':
            angles = np.radians(fixed_angles)
        det_width = (2 * r / np.cos(theta) * (1 + np.sin(theta))) / (det_count - 1) # detector 길이
        proj_geom = astra.create_proj_geom(beam_type, det_width, det_count, angles, source_orign, origin_det)

    if beam_type == 'parallel':
        # source이용(fixed_angles 정의 불필요)
        if use_source == 'yes':
            angles = np.linspace(0, 180, source, False) * (np.pi / 180)  # radian으로 변환

        # angle 고정용(fixed_angles 변수를 정의 해야함)
        if use_source == 'no':
            angles = np.radians(fixed_angles)
        det_width = x_size/(det_count-1)  # detector 길이
        proj_geom = astra.create_proj_geom(beam_type, det_width, det_count, angles)

    vol_geom = astra.create_vol_geom(x_size, y_size)
    proj_id = astra.create_projector(kernal_type, proj_geom, vol_geom)

    # create forward projection
    [sinogram_id, sinogram] = astra.create_sino(phantom_use, proj_id)

    return vol_geom, sinogram_id, sinogram, proj_id


def recon_sart(vol_geom, sinogram_id, iters=100):
    '''
    vol_geom : recon시 phantom 생성 size
    sinogram_id : make_sino에서 생성되는 sinogram_id
    iters : model 반복횟수

    option
    ProjectionOrder : recon시 Projection 투입순서
    sequential, random 택 1
    random선택시 recon할 때마다 racon image 다름
    MinConstraint : recon data 최소값 지정
    MaxConstraint : recon data 최대값 지정
    '''

    recon_id = astra.m.data2d('create', '-vol', vol_geom)
    cfg = astra.astra_dict('SART_CUDA')
    cfg['ProjectionDataId'] = sinogram_id
    cfg['ReconstructionDataId'] = recon_id
    cfg['option'] = {}
    cfg['option']['ProjectionOrder'] = 'sequential'
    cfg['option']['MinConstraint'] = 0
    sart_id = astra.m.algorithm('create', cfg)
    astra.m.algorithm('iterate', sart_id, iters)
    recon = astra.m.data2d('get', recon_id)

    return recon_id, recon
