[Differentiable]
[PreferRecompute]
float3x3 getRotationMatrix(float4 r)
{
    float3x3 R;
    float4 r_n = r / length(r);
    float q0 = r_n[0];
    float q1 = r_n[1];
    float q2 = r_n[2];
    float q3 = r_n[3];
    R[0] = float3(1 - 2 * q2 * q2 - 2 * q3 * q3, 2 * q1 * q2 - 2 * q0 * q3, 2 * q1 * q3 + 2 * q0 * q2);
    R[1] = float3(2 * q1 * q2 + 2 * q0 * q3, 1 - 2 * q1 * q1 - 2 * q3 * q3, 2 * q2 * q3 - 2 * q0 * q1);
    R[2] = float3(2 * q1 * q3 - 2 * q0 * q2, 2 * q2 * q3 + 2 * q0 * q1, 1 - 2 * q1 * q1 - 2 * q2 * q2);
    return R;
}

[Differentiable]
[PreferRecompute]
float3x3 getScalingMatrix(float3 s)
{
    float3x3 T;
    T[0] = float3(s.x, 0, 0);
    T[1] = float3(0, s.y, 0);
    T[2] = float3(0, 0, s.z);
    return T;
}

[Differentiable]
[PreferRecompute]
float2x2 compute_Conv2D(float4 mean, float4x4 viewM, float3x3 Sig)
{
    float4 t = mul(viewM, mean);
    float3x3 J = float3x3(1.0 / t.z, 0, -t.x / (t.z * t.z), 0, 1.0 / t.z, -t.y / (t.z * t.z), 0, 0, 0);
    float3x3 W = float3x3(viewM[0][0], viewM[0][1], viewM[0][2], viewM[1][0], viewM[1][1], viewM[1][2], viewM[2][0], viewM[2][1], viewM[2][2]);
    float3x3 T = mul(J, W);
    float3x3 T_t = transpose(T);
    float3x3 Conv2D_ori = mul(T, mul(Sig, T_t));
    float2x2 Conv2D = float2x2(Conv2D_ori[0][0] + 0.3, Conv2D_ori[0][1], Conv2D_ori[1][0], Conv2D_ori[1][1] + 0.3);
    return Conv2D;
}

[CudaKernel]
[Differentiable]
[AutoPyBindCUDA]
void rasterize(
    DiffTensorView mean,
    DiffTensorView s,
    DiffTensorView r,
    TensorView<float> viewM,
    TensorView<float> projM,
    DiffTensorView color,
    DiffTensorView output)
{
    uint3 globalIdx = cudaBlockIdx() * cudaBlockDim() + cudaThreadIdx();

    if (globalIdx.x > output.size(0) || globalIdx.y > output.size(1))
        return;

    float4 mean_0 = float4(mean[0], mean[1], mean[2], mean[3]);
    float alpha_ori = color[3];
    float3 color_0 = float3(color[0], color[1], color[2]);
    // output.store(uint3(globalIdx.x, globalIdx.y, 0), 0);
    // output.store(uint3(globalIdx.x, globalIdx.y, 1), 0);
    // output.store(uint3(globalIdx.x, globalIdx.y, 2), 0);
    // output.store(uint3(globalIdx.x, globalIdx.y, 3), 0);

    float4 _r = float4(r[0], r[1], r[2], r[3]);
    float3 _s = float3(s[0], s[1], s[2]);
    float3x3 R = getRotationMatrix(_r);
    float3x3 S = getScalingMatrix(_s);
    float3x3 R_t = transpose(R);
    float3x3 S_t = transpose(S);
    float3x3 Sig = mul(R, mul(S, mul(S_t, R_t)));
    // float4x4 _viewM = float4x4(no_diff(viewM[0]), no_diff(viewM[1]), no_diff(viewM[2]), no_diff(viewM[3]));
    // _viewM[0][0] = no_diff(viewM[uint2(0, 0)]);
    float4x4 _viewM;
    //  = float4x4(no_diff(viewM[0]), no_diff(viewM[1]), no_diff(viewM[2]), no_diff(viewM[3]));
    _viewM[0] = float4(no_diff(viewM[uint2(0,0)]), no_diff(viewM[uint2(0,1)]), no_diff(viewM[uint2(0,2)]), no_diff(viewM[uint2(0,3)]));
    _viewM[1] = float4(no_diff(viewM[uint2(1,0)]), no_diff(viewM[uint2(1,1)]), no_diff(viewM[uint2(1,2)]), no_diff(viewM[uint2(1,3)]));
    _viewM[2] = float4(no_diff(viewM[uint2(2,0)]), no_diff(viewM[uint2(2,1)]), no_diff(viewM[uint2(2,2)]), no_diff(viewM[uint2(2,3)]));
    _viewM[3] = float4(no_diff(viewM[uint2(3,0)]), no_diff(viewM[uint2(3,1)]), no_diff(viewM[uint2(3,2)]), no_diff(viewM[uint2(3,3)]));
    float4x4 _projM;
    // _projM[0] = float4(no_diff(projM[uint2(1, 0)]), no_diff(projM[uint2(1, 1)]), no_diff(projM[uint2(1, 2)]), no_diff(projM[uint2(1, 3)]));
    _projM[0] = float4(no_diff(projM[uint2(0,0)]), no_diff(projM[uint2(0,1)]), no_diff(projM[uint2(0,2)]), no_diff(projM[uint2(0,3)]));
    _projM[1] = float4(no_diff(projM[uint2(1,0)]), no_diff(projM[uint2(1,1)]), no_diff(projM[uint2(1,2)]), no_diff(projM[uint2(1,3)]));
    _projM[2] = float4(no_diff(projM[uint2(2,0)]), no_diff(projM[uint2(2,1)]), no_diff(projM[uint2(2,2)]), no_diff(projM[uint2(2,3)]));
    _projM[3] = float4(no_diff(projM[uint2(3,0)]), no_diff(projM[uint2(3,1)]), no_diff(projM[uint2(3,2)]), no_diff(projM[uint2(3,3)]));

    // float4 t = mul(_viewM, mean_0);
    // float3x3 J = float3x3(1.0 / t.z, 0, -t.x / (t.z * t.z), 0, 1.0 / t.z, -t.y / (t.z * t.z), 0, 0, 0);
    // float3x3 W = float3x3(_viewM[0][0], _viewM[0][1], _viewM[0][2], _viewM[1][0], _viewM[1][1], _viewM[1][2], _viewM[2][0], _viewM[2][1], _viewM[2][2]);
    // float3x3 T = mul(J, W);
    // float3x3 T_t = transpose(T);
    // float3x3 Conv2D_ori = mul(T, mul(Sig, T_t));
    // float2x2 Conv2D = float2x2(Conv2D_ori[0][0] + 0.3, Conv2D_ori[0][1], Conv2D_ori[1][0], Conv2D_ori[1][1] + 0.3);

    float2x2 Conv2D = compute_Conv2D(mean_0, _viewM, Sig);

    float det = Conv2D[0][0] * Conv2D[1][1] - Conv2D[0][1] * Conv2D[1][0];
    float inv_det = 1.0 / det;
    float2x2 conic = float2x2(Conv2D[1][1] * inv_det, -Conv2D[0][1] * inv_det, -Conv2D[1][0] * inv_det, Conv2D[0][0] * inv_det);
    float mid = 0.5 * (conic[0][0] + conic[1][1]);
    float lambda1 = mid + sqrt(max(0.1, mid * mid - det));
    float lambda2 = mid - sqrt(max(0.1, mid * mid - det));
    float radius = ceil(3.0 * sqrt(max(lambda1, lambda2)));

    float4 mean_camera = mul(_viewM, mean_0);
    float4 mean_proj = mul(_projM, mean_camera);
    float2 mean_ndc = float2(mean_proj.x / mean_proj.w, mean_proj.y / mean_proj.w);
    float2 mean_screen = float2((mean_ndc.x + 1.0) * 0.5 * output.size(0), (mean_ndc.y + 1) * 0.5 * output.size(1));

    float x_dis = float(globalIdx.x) - mean_screen.x;
    float y_dis = float(globalIdx.y) - mean_screen.y;

    float power_con = -0.5 * (conic[0][0] * x_dis * x_dis + 2 * conic[0][1] * x_dis * y_dis + conic[1][1] * y_dis * y_dis);
    if (power_con > 0)
        return;
    float alpha = min(0.99, alpha_ori * exp(power_con));
    if (alpha < 1.0 / 255.0)
        return;
    float test_t = 1.0 - alpha;
    if (test_t < 0.0001)
        return;

    output.store(uint3(globalIdx.x, globalIdx.y, 0), color_0[0]);
    output.store(uint3(globalIdx.x, globalIdx.y, 1), color_0[1]);
    output.store(uint3(globalIdx.x, globalIdx.y, 2), color_0[2]);
    output.store(uint3(globalIdx.x, globalIdx.y, 3), alpha);

    // if (x_dis > 0 && y_dis > 0) {
    //     output.store(uint3(globalIdx.x, globalIdx.y, 0), 1.0);
    //     output.store(uint3(globalIdx.x, globalIdx.y, 1), 0);
    //     output.store(uint3(globalIdx.x, globalIdx.y, 2), 0);
    //     output.store(uint3(globalIdx.x, globalIdx.y, 3), 1);
    // }else{
    //     output.store(uint3(globalIdx.x, globalIdx.y, 0), 0);
    //     output.store(uint3(globalIdx.x, globalIdx.y, 1), 1.0);
    //     output.store(uint3(globalIdx.x, globalIdx.y, 2), 0);
    //     output.store(uint3(globalIdx.x, globalIdx.y, 3), 1);
    // }
    // if (globalIdx.x == 0) {
    //     // output.store(uint3(globalIdx.x, globalIdx.y, 0), no_diff(viewM[uint2(0,0)]));
    //     // output.store(uint3(globalIdx.x, globalIdx.y, 1), no_diff(viewM[uint2(0,1)]));
    //     // output.store(uint3(globalIdx.x, globalIdx.y, 2), no_diff(viewM[uint2(0,2)]));
    //     // output.store(uint3(globalIdx.x, globalIdx.y, 3), no_diff(viewM[uint2(0,3)]));
    //     // output.store(uint3(globalIdx.x, globalIdx.y, 0), _viewM[0][0]);
    //     // output.store(uint3(globalIdx.x, globalIdx.y, 1), _viewM[0][1]);
    //     // output.store(uint3(globalIdx.x, globalIdx.y, 2), _viewM[0][2]);
    //     // output.store(uint3(globalIdx.x, globalIdx.y, 3), _viewM[0][3]);
    //     output.store(uint3(globalIdx.x, globalIdx.y, 0), mean_camera[0]);
    //     output.store(uint3(globalIdx.x, globalIdx.y, 1), mean_camera[1]);
    //     output.store(uint3(globalIdx.x, globalIdx.y, 2), mean_camera[2]);
    //     output.store(uint3(globalIdx.x, globalIdx.y, 3), mean_camera[3]);
    // } else {
    //     if (globalIdx.x == 1) {
    //         // output.store(uint3(globalIdx.x, globalIdx.y, 0), no_diff(projM[uint2(0,0)]));
    //         // output.store(uint3(globalIdx.x, globalIdx.y, 1), no_diff(projM[uint2(0,1)]));
    //         // output.store(uint3(globalIdx.x, globalIdx.y, 2), no_diff(projM[uint2(0,2)]));
    //         // output.store(uint3(globalIdx.x, globalIdx.y, 3), no_diff(projM[uint2(0,3)]));
    //         output.store(uint3(globalIdx.x, globalIdx.y, 0), mean_proj[0]);
    //         output.store(uint3(globalIdx.x, globalIdx.y, 1), mean_proj[1]);
    //         output.store(uint3(globalIdx.x, globalIdx.y, 2), mean_proj[2]);
    //         output.store(uint3(globalIdx.x, globalIdx.y, 3), mean_proj[3]);
    //     } else {
    //         // output.store(uint3(globalIdx.x, globalIdx.y, 0), no_diff(viewM[uint2(2,0)]));
    //         // output.store(uint3(globalIdx.x, globalIdx.y, 1), no_diff(viewM[uint2(2,1)]));
    //         // output.store(uint3(globalIdx.x, globalIdx.y, 2), no_diff(viewM[uint2(2,2)]));
    //         // output.store(uint3(globalIdx.x, globalIdx.y, 3), no_diff(viewM[uint2(2,3)]));
    //         output.store(uint3(globalIdx.x, globalIdx.y, 0), mean_ndc[0]);
    //         output.store(uint3(globalIdx.x, globalIdx.y, 1), mean_ndc[1]);
    //         output.store(uint3(globalIdx.x, globalIdx.y, 2), mean_screen[0]);
    //         output.store(uint3(globalIdx.x, globalIdx.y, 3), mean_screen[1]);
    //     }
    // }
    return;
}