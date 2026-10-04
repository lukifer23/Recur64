//! Minimal mask/broadcast reproduction, independent of model arithmetic.
use super::*;
use burn::tensor::Bool;

pub fn run<B: AutodiffBackend>(source: &str, device: &B::Device) -> anyhow::Result<Value> {
    let mut cases = vec![];
    for repeat in 0..3 {
        for poison_size in [16, 64, 256, 1024] {
            for fenced in [false, true] {
                for pattern in ["all_valid", "second_row_invalid", "mixed"] {
                    for variant in [
                        "negate_then_expand",
                        "expand_then_negate",
                        "implicit_broadcast",
                        "single_reshape",
                    ] {
                        B::sync(device).map_err(|e| anyhow::anyhow!("{e:?}"))?;
                        let nodes: Vec<bool> = (0..16)
                            .map(|i| match pattern {
                                "second_row_invalid" => i < 8,
                                "mixed" => i % 3 != 0,
                                _ => true,
                            })
                            .collect();
                        let expected_values: Vec<f32> = nodes
                            .iter()
                            .flat_map(|&v| vec![if v { 1.0 } else { 0.0 }; 4 * 256])
                            .collect();
                        let expected_node_count = nodes.iter().filter(|&&v| v).count();
                        let node =
                            Tensor::<B, 2, Bool>::from_data(TensorData::new(nodes, [2, 8]), device);
                        let poison = Tensor::<B, 1>::full([poison_size], 1.0, device);
                        drop(poison);
                        if fenced {
                            B::sync(device).map_err(|e| anyhow::anyhow!("{e:?}"))?;
                        }
                        let input = Tensor::<B, 4>::ones([2, 8, 4, 256], device).require_grad();
                        let mask = match variant {
                            "single_reshape" => node
                                .clone()
                                .bool_not()
                                .reshape([2, 8, 1, 1])
                                .expand([2, 8, 4, 256]),
                            "expand_then_negate" => node
                                .clone()
                                .unsqueeze_dim::<3>(2)
                                .unsqueeze_dim::<4>(3)
                                .expand([2, 8, 4, 256])
                                .bool_not(),
                            "implicit_broadcast" => node
                                .clone()
                                .bool_not()
                                .unsqueeze_dim::<3>(2)
                                .unsqueeze_dim::<4>(3),
                            _ => node
                                .clone()
                                .bool_not()
                                .unsqueeze_dim::<3>(2)
                                .unsqueeze_dim::<4>(3)
                                .expand([2, 8, 4, 256]),
                        };
                        let metadata = format!("{:?}", mask.clone().into_primitive());
                        let output = input.mask_fill(mask.clone(), 0.0).into_data();
                        let expected = TensorData::new(expected_values, [2, 8, 4, 256]);
                        let inverse = mask.into_data();
                        let inverse_values = inverse.iter::<bool>().collect::<Vec<_>>();
                        let node_values = node.into_data().iter::<bool>().collect::<Vec<_>>();
                        cases.push(json!({"repeat":repeat,"pattern":pattern,"expected_node_count":expected_node_count,"poison_size":poison_size,"fenced":fenced,"variant":variant,"mask_metadata":metadata,"node_true_count":node_values.iter().filter(|&&v|v).count(),"inverse_true_count":inverse_values.iter().filter(|&&v|v).count(),"inverse_first_true":inverse_values.iter().position(|&v|v),"output":diff(&Some(output),&Some(expected))?}));
                    }
                }
            }
        }
    }
    let pass = cases
        .iter()
        .all(|c| c["output"]["exact"] == true && c["node_true_count"] == c["expected_node_count"]);
    Ok(
        json!({"schema":"v5_cuda_mask_primitive_v2","stride_preserving_control_exact":cases.iter().filter(|c|c["variant"]=="single_reshape").all(|c|c["output"]["exact"]==true),"source_sha":source,"training_authorized":false,"model_executed":false,"classification":if pass {"MASK_PRIMITIVE_PASS"}else{"MASK_PRIMITIVE_FAIL"},"cases":cases}),
    )
}
