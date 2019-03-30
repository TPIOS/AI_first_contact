from numpy import *
def denoise(im,U_init,tolerance = 0.1, tau=0.125, tv_weight=100):
    """输入：含有噪声的输入图像（灰度图像），U的初始值，TV正则项权值、步长、停业条件
       输出：去早和去除纹理后的图像，纹理残留"""
    m, n = im.shape
    U = U_init
    Px = im
    Py = im
    error = 1

    while (error > tolerance):
        Uold = U
        GradUx = roll(U, -1, axis=1)-U
        GradUy = roll(U, -1, axis=0)-U

        PxNew = Px + (tau/tv_weight)*GradUx
        PyNew = Py + (tau/tv_weight)*GradUy
        NormNew = maximum(1,sqrt(PxNew**2 + PyNew**2))

        Px = PxNew/NormNew
        Py = PyNew/NormNew

        RxPx = roll(Px, 1, axis=1)
        RyPy = roll(Py, 1, axis=0)

        DivP = (Px-RxPx)+(Py-RyPy)
        U = im + tv_weight*DivP

        error = linalg.norm(U-Uold)/sqrt(n*m)

    return U, im-U