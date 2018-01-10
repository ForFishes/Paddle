from __future__ import print_function
import numpy as np
import paddle.v2 as paddle
import paddle.v2.fluid as fluid

BATCH_SIZE = 128
CLIP_MAX = 2e-6
CLIP_MIN = -1e-6

prog = fluid.framework.Program()

with fluid.program_guard(main_program=prog):
    image = fluid.layers.data(name='x', shape=[784], dtype='float32')

    hidden1 = fluid.layers.fc(input=image, size=128, act='relu')
    hidden2 = fluid.layers.fc(input=hidden1, size=64, act='relu')
    predict = fluid.layers.fc(input=hidden2, size=10, act='softmax')

    label = fluid.layers.data(name='y', shape=[1], dtype='int64')

    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(x=cost)

prog_clip = prog.clone()
prog_clip.block(0).var(hidden1.name).set_error_clip(
    fluid.clip.ErrorClipByValue(
        max=CLIP_MAX, min=CLIP_MIN))

avg_cost_clip = prog_clip.block(0).var(avg_cost.name)
fluid.backward.append_backward(loss=avg_cost)
fluid.backward.append_backward(
    loss=avg_cost_clip, callback=fluid.clip.error_clip_callback)

hidden1_grad = prog.block(0).var(hidden1.name + "@GRAD")
hidden1_grad_clip = prog_clip.block(0).var(hidden1.name + "@GRAD")

train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.mnist.train(), buf_size=8192),
    batch_size=BATCH_SIZE)

place = fluid.CPUPlace()
exe = fluid.Executor(place)
feeder = fluid.DataFeeder(feed_list=[image, label], place=place)
exe.run(fluid.default_startup_program())

count = 0
for data in train_reader():
    count += 1
    if count > 5:
        break
    out = exe.run(prog, feed=feeder.feed(data), fetch_list=[hidden1_grad])
    out_clip = exe.run(prog_clip,
                       feed=feeder.feed(data),
                       fetch_list=[hidden1_grad_clip])
    if not (out[0].clip(min=CLIP_MIN, max=CLIP_MAX) == out_clip[0]).all():
        exit(1)

exit(0)
