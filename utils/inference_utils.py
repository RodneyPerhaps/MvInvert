import torch


def run_on_batch(inputs, net, opts, avg_image):
    y_hat, latent = None, None
    result_batch = []
    result_latent = []
    for iter in range(opts.n_iters_per_batch):
        if iter == 0:
            avg_image_for_batch = avg_image.unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)
            x_input = torch.cat([inputs, avg_image_for_batch], dim=1)
        else:
            x_input = torch.cat([inputs, y_hat], dim=1)

        y_hat, latent = net.forward(x_input,
                                    latent=latent,
                                    randomize_noise=False,
                                    return_latents=True,
                                    resize=opts.resize_outputs)

        # store last output
        if iter == opts.n_iters_per_batch - 1:
            for idx in range(inputs.shape[0]):
                result_batch.append(y_hat[idx])
                result_latent.append(latent[idx])

        # resize input to 256 before feeding into next iteration
        y_hat = net.face_pool(y_hat)

    return torch.stack(result_batch), torch.stack(result_latent)
