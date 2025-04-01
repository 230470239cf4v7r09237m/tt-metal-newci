# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import math

HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
BS = ttnn.TensorMemoryLayout.BLOCK_SHARDED
WS = ttnn.TensorMemoryLayout.WIDTH_SHARDED


class Yolov6x_Conv2D:
    def __init__(
        self,
        conv,
        conv_pth,
        bn=None,
        device=None,
        cache={},
        activation="",
        activation_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat8_b,
        use_1d_systolic_array=True,
        use_shallow_conv_variant=False,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        auto_shard=False,
        is_nhw_c=False,
        is_nhwc=False,
    ):
        self.is_nhw_c = is_nhw_c
        self.is_nhwc = is_nhwc
        self.conv = conv
        self.device = device
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.padding = conv.padding
        self.stride = conv.stride
        self.groups = conv.groups
        self.use_1d_systolic_array = use_1d_systolic_array
        self.deallocate_activation = False
        self.cache = cache
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
            math_approx_mode=True,
        )

        self.conv_config = ttnn.Conv2dConfig(
            dtype=activation_dtype,
            weights_dtype=weights_dtype,
            shard_layout=shard_layout if not auto_shard else None,
            deallocate_activation=self.deallocate_activation,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True if self.use_1d_systolic_array else False,
            activation=activation,
            input_channels_alignment=8 if use_shallow_conv_variant and not auto_shard else 32,
        )
        config_override = None
        if config_override and "act_block_h" in config_override and not auto_shard:
            self.conv_config.act_block_h_override = config_override["act_block_h"]

        if "bias" in conv_pth:
            bias = ttnn.from_device(conv_pth.bias)
            self.bias = bias
        else:
            self.bias = None

        weight = ttnn.from_device(conv_pth.weight)
        self.weight = weight

    def __call__(self, x):
        if self.is_nhw_c:
            input_height = int(math.sqrt(x.shape[2]))
            input_width = int(math.sqrt(x.shape[2]))
            batch_size = x.shape[0]
        elif self.is_nhwc:
            input_height = x.shape[1]
            input_width = x.shape[2]
            batch_size = x.shape[0]
        else:
            batch_size = x.shape[0]  # self.conv.batch_size
            input_height = self.conv.input_height
            input_width = self.conv.input_width

        [x, [output_height, output_width], [self.weight, self.bias]] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            input_height=input_height,
            input_width=input_width,
            batch_size=batch_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            conv_config=self.conv_config,
            conv_op_cache=self.cache,
            groups=self.groups,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        return x


class Yolov6x_Conv_T_2D:
    def __init__(
        self,
        conv,
        conv_pth,
        device=None,
        cache={},
        activations_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat8_b,
        use_1d_systolic_array=True,
        shard_layout=None,
        auto_shard=False,
        use_shallow_conv_variant=False,
        config_override=None,
    ):
        self.input_channels = conv.in_channels
        self.output_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.padding = conv.padding
        self.stride = conv.stride
        self.device = device
        if shard_layout is None and not auto_shard:
            shard_layout = (
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED
                if use_1d_systolic_array
                else ttnn.TensorMemoryLayout.BLOCK_SHARDED
            )
        self.conv_config = ttnn.Conv2dConfig(
            dtype=activations_dtype,
            weights_dtype=weights_dtype,
            shard_layout=shard_layout,
            input_channels_alignment=(
                16 if use_shallow_conv_variant or (self.input_channels == 16 and self.input_height == 115) else 32
            ),
            deallocate_activation=False,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            output_layout=ttnn.TILE_LAYOUT,
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        if config_override and "act_block_h" in config_override:
            self.conv_config.act_block_h_override = config_override["act_block_h"]

        if "bias" in conv_pth.conv_t:
            bias = ttnn.from_device(conv_pth.conv_t.bias)
            self.bias = bias
        else:
            self.bias = None

        self.weight = ttnn.from_device(conv_pth.conv_t.weight)

    def __call__(self, x):
        input_height = int(math.sqrt(x.shape[2]))
        input_width = int(math.sqrt(x.shape[2]))
        batch_size = x.shape[0]

        [tt_output_tensor_on_device, [out_height, out_width], [weights_device, bias_device]] = ttnn.conv_transpose2d(
            input_tensor=x,
            weight_tensor=self.weight,
            in_channels=self.input_channels,
            out_channels=self.output_channels,
            device=self.device,
            bias_tensor=self.bias,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=(0, 0),
            output_padding=(0, 0),
            dilation=(1, 1),
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            groups=1,
            mirror_kernel=True,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        return tt_output_tensor_on_device


def sharded_concat(input_tensors, num_cores=64, dim=3):  # expected input tensors to be in fp16, RM, same (h*w)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
    in_shard_width = input_tensors[0].shape[-1]
    shard_height = (input_tensors[0].shape[2] + num_cores - 1) // num_cores
    input_sharded_memory_config = ttnn.create_sharded_memory_config(
        (shard_height, in_shard_width),
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    out_shard_width = 0
    for i in range(len(input_tensors)):
        out_shard_width += input_tensors[i].shape[-1]
        input_tensors[i] = ttnn.to_memory_config(input_tensors[i], input_sharded_memory_config)
    output_sharded_memory_config = ttnn.create_sharded_memory_config(
        (shard_height, out_shard_width),
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    output = ttnn.concat(input_tensors, dim, memory_config=output_sharded_memory_config)
    output = ttnn.sharded_to_interleaved(output, memory_config=ttnn.L1_MEMORY_CONFIG)

    return output


class Ttnn_Sppf:
    def __init__(self, device, parameter, model_params):
        self.parameter = parameter
        self.model_params = model_params
        self.cv1 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.cv1.conv,
            conv_pth=parameter.cv1.conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhwc=True,
        )
        self.cv2 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.cv2.conv,
            conv_pth=parameter.cv2.conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )

    def __call__(self, device, x):
        x = self.cv1(x)
        if x.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x1 = x
        x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        m1 = ttnn.max_pool2d(
            x,
            batch_size=x.shape[0],
            input_h=int(math.sqrt(x.shape[2])),
            input_w=int(math.sqrt(x.shape[2])),
            channels=320,
            kernel_size=[5, 5],
            stride=[1, 1],
            padding=[2, 2],
            dilation=[1, 1],
        )
        m2 = ttnn.max_pool2d(
            m1,
            batch_size=m1.shape[0],
            input_h=int(math.sqrt(m1.shape[2])),
            input_w=int(math.sqrt(m1.shape[2])),
            channels=320,
            kernel_size=[5, 5],
            stride=[1, 1],
            padding=[2, 2],
            dilation=[1, 1],
        )
        m3 = ttnn.max_pool2d(
            m2,
            batch_size=m2.shape[0],
            input_h=int(math.sqrt(m2.shape[2])),
            input_w=int(math.sqrt(m2.shape[2])),
            channels=320,
            kernel_size=[5, 5],
            stride=[1, 1],
            padding=[2, 2],
            dilation=[1, 1],
        )
        use_sharded_concat = True
        if use_sharded_concat:
            y = sharded_concat([x1, m1, m2, m3])
        else:
            y = ttnn.concat([x1, m1, m2, m3], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = self.cv2(y)
        ttnn.deallocate(x1)
        ttnn.deallocate(m1)
        ttnn.deallocate(m2)
        ttnn.deallocate(m3)
        return x


class Ttnn_Detect:
    def __init__(self, device, parameter, model_params):
        self.cv2_0_0 = Yolov6x_Conv2D(
            model_params.cv2[0][0].conv,
            parameter.cv2[0][0].conv,
            auto_shard=True,
            shard_layout=None,
            activation="relu",
            device=device,
            is_nhw_c=True,
        )
        self.cv2_0_1 = Yolov6x_Conv2D(
            model_params.cv2[0][1].conv,
            parameter.cv2[0][1].conv,
            auto_shard=True,
            shard_layout=None,
            activation="relu",
            device=device,
            is_nhw_c=True,
        )  # BS
        self.cv2_0_2 = Yolov6x_Conv2D(
            model_params.cv2[0][2],
            parameter.cv2[0][2],
            auto_shard=True,
            shard_layout=None,
            device=device,
            is_nhw_c=True,
        )  # BS

        self.cv2_1_0 = Yolov6x_Conv2D(
            model_params.cv2[1][0].conv,
            parameter.cv2[1][0].conv,
            auto_shard=True,
            shard_layout=None,
            activation="relu",
            device=device,
            is_nhw_c=True,
        )
        self.cv2_1_1 = Yolov6x_Conv2D(
            model_params.cv2[1][1].conv,
            parameter.cv2[1][1].conv,
            auto_shard=True,
            shard_layout=None,
            activation="relu",
            device=device,
            is_nhw_c=True,
        )  # BS
        self.cv2_1_2 = Yolov6x_Conv2D(
            model_params.cv2[1][2],
            parameter.cv2[1][2],
            auto_shard=True,
            shard_layout=None,
            device=device,
            is_nhw_c=True,
        )  # BS

        self.cv2_2_0 = Yolov6x_Conv2D(
            model_params.cv2[2][0].conv,
            parameter.cv2[2][0].conv,
            auto_shard=True,
            shard_layout=None,
            activation="relu",
            device=device,
            is_nhw_c=True,
        )
        self.cv2_2_1 = Yolov6x_Conv2D(
            model_params.cv2[2][1].conv,
            parameter.cv2[2][1].conv,
            auto_shard=True,
            shard_layout=None,
            activation="relu",
            device=device,
            is_nhw_c=True,
        )  # BS
        self.cv2_2_2 = Yolov6x_Conv2D(
            model_params.cv2[2][2],
            parameter.cv2[2][2],
            auto_shard=True,
            shard_layout=None,
            device=device,
            is_nhw_c=True,
        )  # BS

        self.cv3_0_0 = Yolov6x_Conv2D(
            model_params.cv3[0][0].conv,
            parameter.cv3[0][0].conv,
            auto_shard=True,
            shard_layout=None,
            activation="relu",
            device=device,
            is_nhw_c=True,
        )
        self.cv3_0_1 = Yolov6x_Conv2D(
            model_params.cv3[0][1].conv,
            parameter.cv3[0][1].conv,
            auto_shard=True,
            shard_layout=None,
            activation="relu",
            device=device,
            is_nhw_c=True,
        )  # BS
        self.cv3_0_2 = Yolov6x_Conv2D(
            model_params.cv3[0][2],
            parameter.cv3[0][2],
            auto_shard=True,
            shard_layout=None,
            device=device,
            is_nhw_c=True,
        )  # BS

        self.cv3_1_0 = Yolov6x_Conv2D(
            model_params.cv3[1][0].conv,
            parameter.cv3[1][0].conv,
            auto_shard=True,
            shard_layout=None,
            activation="relu",
            device=device,
            is_nhw_c=True,
        )
        self.cv3_1_1 = Yolov6x_Conv2D(
            model_params.cv3[1][1].conv,
            parameter.cv3[1][1].conv,
            auto_shard=True,
            shard_layout=None,
            activation="relu",
            device=device,
            is_nhw_c=True,
        )  # BS
        self.cv3_1_2 = Yolov6x_Conv2D(
            model_params.cv3[1][2],
            parameter.cv3[1][2],
            auto_shard=True,
            shard_layout=None,
            device=device,
            is_nhw_c=True,
        )  # BS

        self.cv3_2_0 = Yolov6x_Conv2D(
            model_params.cv3[2][0].conv,
            parameter.cv3[2][0].conv,
            auto_shard=True,
            shard_layout=None,
            activation="relu",
            device=device,
            is_nhw_c=True,
        )
        self.cv3_2_1 = Yolov6x_Conv2D(
            model_params.cv3[2][1].conv,
            parameter.cv3[2][1].conv,
            auto_shard=True,
            shard_layout=None,
            activation="relu",
            device=device,
            is_nhw_c=True,
        )  # BS
        self.cv3_2_2 = Yolov6x_Conv2D(
            model_params.cv3[2][2],
            parameter.cv3[2][2],
            auto_shard=True,
            shard_layout=None,
            device=device,
            is_nhw_c=True,
        )  # BS

        self.dfl = Yolov6x_Conv2D(
            model_params.dfl.conv, parameter.dfl.conv, auto_shard=True, shard_layout=None, device=device, is_nhw_c=True
        )
        self.anchors = parameter.anchors
        self.strides = parameter.strides

    def __call__(self, device, y1, y2, y3):
        x1 = self.cv2_0_0(y1)
        x1 = self.cv2_0_1(x1)
        x1 = self.cv2_0_2(x1)

        x2 = self.cv2_1_0(y2)
        x2 = self.cv2_1_1(x2)
        x2 = self.cv2_1_2(x2)

        x3 = self.cv2_2_0(y3)
        x3 = self.cv2_2_1(x3)
        x3 = self.cv2_2_2(x3)

        x4 = self.cv3_0_0(y1)
        x4 = self.cv3_0_1(x4)
        x4 = self.cv3_0_2(x4)

        x5 = self.cv3_1_0(y2)
        x5 = self.cv3_1_1(x5)
        x5 = self.cv3_1_2(x5)

        x6 = self.cv3_2_0(y3)
        x6 = self.cv3_2_1(x6)
        x6 = self.cv3_2_2(x6)

        x1 = ttnn.sharded_to_interleaved(x1, memory_config=ttnn.L1_MEMORY_CONFIG)
        x2 = ttnn.sharded_to_interleaved(x2, memory_config=ttnn.L1_MEMORY_CONFIG)
        x3 = ttnn.sharded_to_interleaved(x3, memory_config=ttnn.L1_MEMORY_CONFIG)
        x4 = ttnn.sharded_to_interleaved(x4, memory_config=ttnn.L1_MEMORY_CONFIG)
        x5 = ttnn.sharded_to_interleaved(x5, memory_config=ttnn.L1_MEMORY_CONFIG)
        x6 = ttnn.sharded_to_interleaved(x6, memory_config=ttnn.L1_MEMORY_CONFIG)

        y1 = ttnn.concat((x1, x4), -1, memory_config=ttnn.L1_MEMORY_CONFIG)
        y2 = ttnn.concat((x2, x5), -1, memory_config=ttnn.L1_MEMORY_CONFIG)
        y3 = ttnn.concat((x3, x6), -1, memory_config=ttnn.L1_MEMORY_CONFIG)

        y = ttnn.concat((y1, y2, y3), dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
        y = ttnn.squeeze(y, dim=0)

        ya, yb = y[:, :, :64], y[:, :, 64:144]
        ttnn.deallocate(y1)
        ttnn.deallocate(y2)
        ttnn.deallocate(y3)
        ttnn.deallocate(x1)
        ttnn.deallocate(x2)
        ttnn.deallocate(x3)
        ttnn.deallocate(x4)
        ttnn.deallocate(x5)
        ttnn.deallocate(x6)
        ttnn.deallocate(y)

        ya = ttnn.reshape(ya, (ya.shape[0], y.shape[1], 4, 16))
        ya = ttnn.permute(ya, (0, 2, 1, 3))  # 0.999
        ya = ttnn.softmax(ya, dim=-1, numeric_stable=True)  # 0.9949745397952091
        c = self.dfl(ya)  # 0.9654762051555557 (0.9654096135662267 after auto shard)

        ttnn.deallocate(ya)
        c = ttnn.sharded_to_interleaved(c, memory_config=ttnn.L1_MEMORY_CONFIG)
        c = ttnn.to_layout(c, layout=ttnn.ROW_MAJOR_LAYOUT)
        c = ttnn.permute(c, (0, 3, 1, 2))
        c = ttnn.reshape(c, (c.shape[0], 1, 4, int(c.shape[3] / 4)))
        c = ttnn.reshape(c, (c.shape[0], c.shape[1] * c.shape[2], c.shape[3]))

        c1, c2 = c[:, :2, :], c[:, 2:4, :]
        anchor, strides = self.anchors, self.strides
        anchor = ttnn.to_memory_config(anchor, memory_config=ttnn.L1_MEMORY_CONFIG)
        strides = ttnn.to_memory_config(strides, memory_config=ttnn.L1_MEMORY_CONFIG)
        c1 = ttnn.to_layout(c1, layout=ttnn.TILE_LAYOUT)
        c2 = ttnn.to_layout(c2, layout=ttnn.TILE_LAYOUT)

        c1 = anchor - c1  # 0.999
        c2 = anchor + c2  # 0.999
        z2 = c1 + c2
        z2 = ttnn.div(z2, 2)  # 0.999
        z1 = c2 - c1  # 0.7676696715044726
        z = ttnn.concat((z2, z1), dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)  # 0.999
        z = ttnn.multiply(z, strides)

        yb = ttnn.permute(yb, (0, 2, 1))
        yb = ttnn.sigmoid(yb)  # 0.9987866613751747

        ttnn.deallocate(c)
        ttnn.deallocate(z1)
        ttnn.deallocate(z2)
        ttnn.deallocate(c1)
        ttnn.deallocate(c2)
        ttnn.deallocate(anchor)
        ttnn.deallocate(strides)

        z = ttnn.reallocate(z)
        yb = ttnn.reallocate(yb)

        z = ttnn.to_layout(z, layout=ttnn.ROW_MAJOR_LAYOUT)
        yb = ttnn.to_layout(yb, layout=ttnn.ROW_MAJOR_LAYOUT)
        out = ttnn.concat((z, yb), dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)  # 0.999

        ttnn.deallocate(yb)
        ttnn.deallocate(z)
        return out


class Ttnn_Yolov6x:
    def __init__(self, device, parameter, model_params):
        self.parameter = parameter
        self.model_params = model_params
        self.conv0 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[0].conv,
            conv_pth=parameter.model[0].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhwc=True,
        )
        self.conv1 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[1].conv,
            conv_pth=parameter.model[1].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )

        self.conv2_0 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[2][0].conv,
            conv_pth=parameter.model[2][0].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv2_1 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[2][1].conv,
            conv_pth=parameter.model[2][1].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv2_2 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[2][2].conv,
            conv_pth=parameter.model[2][2].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv2_3 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[2][3].conv,
            conv_pth=parameter.model[2][3].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv2_4 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[2][4].conv,
            conv_pth=parameter.model[2][4].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv2_5 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[2][5].conv,
            conv_pth=parameter.model[2][5].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )

        self.conv3 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[3].conv,
            conv_pth=parameter.model[3].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )

        self.conv4_0 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[4][0].conv,
            conv_pth=parameter.model[4][0].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv4_1 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[4][1].conv,
            conv_pth=parameter.model[4][1].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv4_2 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[4][2].conv,
            conv_pth=parameter.model[4][2].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv4_3 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[4][3].conv,
            conv_pth=parameter.model[4][3].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv4_4 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[4][4].conv,
            conv_pth=parameter.model[4][4].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv4_5 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[4][5].conv,
            conv_pth=parameter.model[4][5].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv4_6 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[4][6].conv,
            conv_pth=parameter.model[4][6].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv4_7 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[4][7].conv,
            conv_pth=parameter.model[4][7].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv4_8 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[4][8].conv,
            conv_pth=parameter.model[4][8].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv4_9 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[4][9].conv,
            conv_pth=parameter.model[4][9].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv4_10 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[4][10].conv,
            conv_pth=parameter.model[4][10].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv4_11 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[4][11].conv,
            conv_pth=parameter.model[4][11].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )

        self.conv5 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[5].conv,
            conv_pth=parameter.model[5].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )

        self.conv6_0 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[6][0].conv,
            conv_pth=parameter.model[6][0].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv6_1 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[6][1].conv,
            conv_pth=parameter.model[6][1].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv6_2 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[6][2].conv,
            conv_pth=parameter.model[6][2].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv6_3 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[6][3].conv,
            conv_pth=parameter.model[6][3].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv6_4 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[6][4].conv,
            conv_pth=parameter.model[6][4].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv6_5 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[6][5].conv,
            conv_pth=parameter.model[6][5].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv6_6 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[6][6].conv,
            conv_pth=parameter.model[6][6].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv6_7 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[6][7].conv,
            conv_pth=parameter.model[6][7].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv6_8 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[6][8].conv,
            conv_pth=parameter.model[6][8].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv6_9 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[6][9].conv,
            conv_pth=parameter.model[6][9].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv6_10 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[6][10].conv,
            conv_pth=parameter.model[6][10].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv6_11 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[6][11].conv,
            conv_pth=parameter.model[6][11].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv6_12 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[6][12].conv,
            conv_pth=parameter.model[6][12].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv6_13 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[6][13].conv,
            conv_pth=parameter.model[6][13].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv6_14 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[6][14].conv,
            conv_pth=parameter.model[6][14].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv6_15 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[6][15].conv,
            conv_pth=parameter.model[6][15].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv6_16 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[6][16].conv,
            conv_pth=parameter.model[6][16].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv6_17 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[6][17].conv,
            conv_pth=parameter.model[6][17].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )

        self.conv7 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[7].conv,
            conv_pth=parameter.model[7].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )

        self.conv8_0 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[8][0].conv,
            conv_pth=parameter.model[8][0].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv8_1 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[8][1].conv,
            conv_pth=parameter.model[8][1].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv8_2 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[8][2].conv,
            conv_pth=parameter.model[8][2].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv8_3 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[8][3].conv,
            conv_pth=parameter.model[8][3].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv8_4 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[8][4].conv,
            conv_pth=parameter.model[8][4].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv8_5 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[8][5].conv,
            conv_pth=parameter.model[8][5].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )

        self.sppf = Ttnn_Sppf(device, parameter.model[9], model_params.model[9])

        self.conv10 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[10].conv,
            conv_pth=parameter.model[10].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )

        self.conv_t_11 = Yolov6x_Conv_T_2D(
            model_params.model[11],
            parameter.model[11],
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            device=device,
        )

        self.conv13 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[13].conv,
            conv_pth=parameter.model[13].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )

        self.conv14_0 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[14][0].conv,
            conv_pth=parameter.model[14][0].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv14_1 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[14][1].conv,
            conv_pth=parameter.model[14][1].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv14_2 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[14][2].conv,
            conv_pth=parameter.model[14][2].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv14_3 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[14][3].conv,
            conv_pth=parameter.model[14][3].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv14_4 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[14][4].conv,
            conv_pth=parameter.model[14][4].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv14_5 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[14][5].conv,
            conv_pth=parameter.model[14][5].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv14_6 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[14][6].conv,
            conv_pth=parameter.model[14][6].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv14_7 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[14][7].conv,
            conv_pth=parameter.model[14][7].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv14_8 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[14][8].conv,
            conv_pth=parameter.model[14][8].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )

        self.conv15 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[15].conv,
            conv_pth=parameter.model[15].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )

        self.conv_t_16 = Yolov6x_Conv_T_2D(
            model_params.model[16],
            parameter.model[16],
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            device=device,
        )

        self.conv18 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[18].conv,
            conv_pth=parameter.model[18].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )

        self.conv19_0 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[19][0].conv,
            conv_pth=parameter.model[19][0].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv19_1 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[19][1].conv,
            conv_pth=parameter.model[19][1].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv19_2 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[19][2].conv,
            conv_pth=parameter.model[19][2].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv19_3 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[19][3].conv,
            conv_pth=parameter.model[19][3].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv19_4 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[19][4].conv,
            conv_pth=parameter.model[19][4].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv19_5 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[19][5].conv,
            conv_pth=parameter.model[19][5].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv19_6 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[19][6].conv,
            conv_pth=parameter.model[19][6].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv19_7 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[19][7].conv,
            conv_pth=parameter.model[19][7].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv19_8 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[19][8].conv,
            conv_pth=parameter.model[19][8].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )

        self.conv20 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[20].conv,
            conv_pth=parameter.model[20].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv22 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[22].conv,
            conv_pth=parameter.model[22].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )

        self.conv23_0 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[23][0].conv,
            conv_pth=parameter.model[23][0].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv23_1 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[23][1].conv,
            conv_pth=parameter.model[23][1].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv23_2 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[23][2].conv,
            conv_pth=parameter.model[23][2].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv23_3 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[23][3].conv,
            conv_pth=parameter.model[23][3].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv23_4 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[23][4].conv,
            conv_pth=parameter.model[23][4].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv23_5 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[23][5].conv,
            conv_pth=parameter.model[23][5].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv23_6 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[23][6].conv,
            conv_pth=parameter.model[23][6].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv23_7 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[23][7].conv,
            conv_pth=parameter.model[23][7].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv23_8 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[23][8].conv,
            conv_pth=parameter.model[23][8].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )

        self.conv24 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[24].conv,
            conv_pth=parameter.model[24].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv26 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[26].conv,
            conv_pth=parameter.model[26].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )

        self.conv27_0 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[27][0].conv,
            conv_pth=parameter.model[27][0].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv27_1 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[27][1].conv,
            conv_pth=parameter.model[27][1].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv27_2 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[27][2].conv,
            conv_pth=parameter.model[27][2].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv27_3 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[27][3].conv,
            conv_pth=parameter.model[27][3].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv27_4 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[27][4].conv,
            conv_pth=parameter.model[27][4].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv27_5 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[27][5].conv,
            conv_pth=parameter.model[27][5].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv27_6 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[27][6].conv,
            conv_pth=parameter.model[27][6].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv27_7 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[27][7].conv,
            conv_pth=parameter.model[27][7].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )
        self.conv27_8 = Yolov6x_Conv2D(
            device=device,
            conv=model_params.model[27][8].conv,
            conv_pth=parameter.model[27][8].conv,
            shard_layout=None,
            auto_shard=True,
            activation="relu",
            is_nhw_c=True,
        )

        self.detect = Ttnn_Detect(device, parameter.model[28], model_params.model[28])

    def __call__(self, device, x):
        x0 = self.conv0(x)  # 0.9975022195673564
        x1 = self.conv1(x0)  # 0.999

        x2_0 = self.conv2_0(x1)  # 0.999
        x2_1 = self.conv2_1(x2_0)  # 0.9217840283448264
        x2_2 = self.conv2_2(x2_1)
        x2_3 = self.conv2_3(x2_2)
        x2_4 = self.conv2_4(x2_3)
        x2_5 = self.conv2_5(x2_4)

        ttnn.deallocate(x0)
        ttnn.deallocate(x1)
        ttnn.deallocate(x2_0)
        ttnn.deallocate(x2_1)
        ttnn.deallocate(x2_2)
        ttnn.deallocate(x2_3)
        ttnn.deallocate(x2_4)

        x3 = self.conv3(x2_5)

        x4_0 = self.conv4_0(x3)
        x4_1 = self.conv4_1(x4_0)
        x4_2 = self.conv4_2(x4_1)
        x4_3 = self.conv4_3(x4_2)
        x4_4 = self.conv4_4(x4_3)
        x4_5 = self.conv4_5(x4_4)
        x4_6 = self.conv4_6(x4_5)
        x4_7 = self.conv4_7(x4_6)
        x4_8 = self.conv4_8(x4_7)
        x4_9 = self.conv4_9(x4_8)
        x4_10 = self.conv4_10(x4_9)
        x4_11 = self.conv4_11(x4_10)

        x5 = self.conv5(x4_11)

        ttnn.deallocate(x2_5)
        ttnn.deallocate(x3)
        ttnn.deallocate(x4_0)
        ttnn.deallocate(x4_1)
        ttnn.deallocate(x4_2)
        ttnn.deallocate(x4_3)
        ttnn.deallocate(x4_4)
        ttnn.deallocate(x4_5)
        ttnn.deallocate(x4_6)
        ttnn.deallocate(x4_7)
        ttnn.deallocate(x4_8)
        ttnn.deallocate(x4_9)
        ttnn.deallocate(x4_10)

        x6_0 = self.conv6_0(x5)
        x6_1 = self.conv6_1(x6_0)
        x6_2 = self.conv6_2(x6_1)
        x6_3 = self.conv6_3(x6_2)
        x6_4 = self.conv6_4(x6_3)
        x6_5 = self.conv6_5(x6_4)
        x6_6 = self.conv6_6(x6_5)
        x6_7 = self.conv6_7(x6_6)
        x6_8 = self.conv6_8(x6_7)
        x6_9 = self.conv6_9(x6_8)
        x6_10 = self.conv6_10(x6_9)
        x6_11 = self.conv6_11(x6_10)
        x6_12 = self.conv6_12(x6_11)
        x6_13 = self.conv6_13(x6_12)
        x6_14 = self.conv6_14(x6_13)
        x6_15 = self.conv6_15(x6_14)
        x6_16 = self.conv6_16(x6_15)
        x6_17 = self.conv6_17(x6_16)  # 0.9796952769406674

        ttnn.deallocate(x5)
        ttnn.deallocate(x6_0)
        ttnn.deallocate(x6_1)
        ttnn.deallocate(x6_2)
        ttnn.deallocate(x6_3)
        ttnn.deallocate(x6_4)
        ttnn.deallocate(x6_5)
        ttnn.deallocate(x6_6)
        ttnn.deallocate(x6_7)
        ttnn.deallocate(x6_8)
        ttnn.deallocate(x6_9)
        ttnn.deallocate(x6_10)
        ttnn.deallocate(x6_11)
        ttnn.deallocate(x6_12)
        ttnn.deallocate(x6_13)
        ttnn.deallocate(x6_14)
        ttnn.deallocate(x6_15)
        ttnn.deallocate(x6_16)

        x7 = self.conv7(x6_17)

        x8_0 = self.conv8_0(x7)
        x8_1 = self.conv8_1(x8_0)
        x8_2 = self.conv8_2(x8_1)
        x8_3 = self.conv8_3(x8_2)
        x8_4 = self.conv8_4(x8_3)
        x8_5 = self.conv8_5(x8_4)

        ttnn.deallocate(x7)
        ttnn.deallocate(x8_0)
        ttnn.deallocate(x8_1)
        ttnn.deallocate(x8_2)
        ttnn.deallocate(x8_3)
        ttnn.deallocate(x8_4)

        x9 = self.sppf(device, x8_5)

        x10 = self.conv10(x9)  # 0.9923337476011245

        x11 = self.conv_t_11(x10)  # 0.999

        ttnn.deallocate(x8_5)
        ttnn.deallocate(x9)

        x11 = ttnn.sharded_to_interleaved(x11, ttnn.L1_MEMORY_CONFIG)
        x12 = ttnn.concat([x11, x6_17], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)

        x13 = self.conv13(x12)

        ttnn.deallocate(x11)
        ttnn.deallocate(x12)

        x14_0 = self.conv14_0(x13)
        x14_1 = self.conv14_1(x14_0)
        x14_2 = self.conv14_2(x14_1)
        x14_3 = self.conv14_3(x14_2)
        x14_4 = self.conv14_4(x14_3)
        x14_5 = self.conv14_5(x14_4)
        x14_6 = self.conv14_6(x14_5)
        x14_7 = self.conv14_7(x14_6)
        x14_8 = self.conv14_8(x14_7)  # 0.9970819747146192

        ttnn.deallocate(x13)
        ttnn.deallocate(x14_0)
        ttnn.deallocate(x14_1)
        ttnn.deallocate(x14_2)
        ttnn.deallocate(x14_3)
        ttnn.deallocate(x14_4)
        ttnn.deallocate(x14_5)
        ttnn.deallocate(x14_6)
        ttnn.deallocate(x14_7)

        x15 = self.conv15(x14_8)
        x16 = self.conv_t_16(x15)

        ttnn.deallocate(x14_8)

        x16 = ttnn.sharded_to_interleaved(x16, ttnn.L1_MEMORY_CONFIG)
        x17 = ttnn.concat([x16, x4_11], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        x18 = self.conv18(x17)  # 0.999

        ttnn.deallocate(x16)
        ttnn.deallocate(x17)
        ttnn.deallocate(x4_11)

        x19_0 = self.conv19_0(x18)
        x19_1 = self.conv19_1(x19_0)
        x19_2 = self.conv19_2(x19_1)
        x19_3 = self.conv19_3(x19_2)
        x19_4 = self.conv19_4(x19_3)
        x19_5 = self.conv19_5(x19_4)
        x19_6 = self.conv19_6(x19_5)
        x19_7 = self.conv19_7(x19_6)
        x19_8 = self.conv19_8(x19_7)

        ttnn.deallocate(x18)
        ttnn.deallocate(x19_0)
        ttnn.deallocate(x19_1)
        ttnn.deallocate(x19_2)
        ttnn.deallocate(x19_3)
        ttnn.deallocate(x19_4)
        ttnn.deallocate(x19_5)
        ttnn.deallocate(x19_6)
        ttnn.deallocate(x19_7)

        x20 = self.conv20(x19_8)

        x21 = ttnn.concat([x20, x15], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        x22 = self.conv22(x21)  # 0.996656566141146

        ttnn.deallocate(x20)
        ttnn.deallocate(x15)
        ttnn.deallocate(x21)

        x23_0 = self.conv23_0(x22)
        x23_1 = self.conv23_1(x23_0)
        x23_2 = self.conv23_2(x23_1)
        x23_3 = self.conv23_3(x23_2)
        x23_4 = self.conv23_4(x23_3)
        x23_5 = self.conv23_5(x23_4)
        x23_6 = self.conv23_6(x23_5)
        x23_7 = self.conv23_7(x23_6)
        x23_8 = self.conv23_8(x23_7)

        ttnn.deallocate(x22)
        ttnn.deallocate(x23_0)
        ttnn.deallocate(x23_1)
        ttnn.deallocate(x23_2)
        ttnn.deallocate(x23_3)
        ttnn.deallocate(x23_4)
        ttnn.deallocate(x23_5)
        ttnn.deallocate(x23_6)
        ttnn.deallocate(x23_7)

        x24 = self.conv24(x23_8)

        x10 = ttnn.from_device(x10)
        x10 = ttnn.to_dtype(x10, dtype=ttnn.bfloat8_b)
        x10 = ttnn.to_device(x10, device)

        x25 = ttnn.concat([x24, x10], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        x26 = self.conv26(x25)

        ttnn.deallocate(x24)
        ttnn.deallocate(x25)
        ttnn.deallocate(x10)

        x27_0 = self.conv27_0(x26)
        x27_1 = self.conv27_1(x27_0)
        x27_2 = self.conv27_2(x27_1)
        x27_3 = self.conv27_3(x27_2)
        x27_4 = self.conv27_4(x27_3)
        x27_5 = self.conv27_5(x27_4)
        x27_6 = self.conv27_6(x27_5)
        x27_7 = self.conv27_7(x27_6)
        x27_8 = self.conv27_8(x27_7)  # 0.9937663709395512

        ttnn.deallocate(x26)
        ttnn.deallocate(x27_0)
        ttnn.deallocate(x27_1)
        ttnn.deallocate(x27_2)
        ttnn.deallocate(x27_3)
        ttnn.deallocate(x27_4)
        ttnn.deallocate(x27_5)
        ttnn.deallocate(x27_6)
        ttnn.deallocate(x27_7)

        x28 = self.detect(device, x19_8, x23_8, x27_8)

        ttnn.deallocate(x19_8)
        ttnn.deallocate(x23_8)
        ttnn.deallocate(x27_8)

        return x28
