from pi_gan_pytorch import piGAN, Trainer

gan = piGAN(
    image_size = 128,
    dim = 512
).cuda()

trainer = Trainer(
    gan = gan,
    folder = '/media/jasonperhaps/D1/Dataset/DFR1024'
)

trainer()

