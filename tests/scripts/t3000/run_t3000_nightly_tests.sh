#!/bin/bash
set -eo pipefail

run_t3000_llama3.2-90b-vision_nightly_tests() {
  # Record the start time
  fail=0
  start_time=$(date +%s)

  echo "LOG_METAL: Running run_t3000_llama3.2-90b-vision_nightly_tests"

  wh_arch_yaml=wormhole_b0_80_arch_eth_dispatch.yaml
  llama90b=/mnt/MLPerf/tt_dnn-models/llama/Llama3.2-90B-Vision-Instruct/
  LLAMA_DIR=$llama90b WH_ARCH_YAML=$wh_arch_yaml pytest models/tt_transformers/tests/multimodal/test_llama_cross_attention_transformer_text.py --timeout 18000 ; fail+=$?
  LLAMA_DIR=$llama90b WH_ARCH_YAML=$wh_arch_yaml pytest models/tt_transformers/tests/test_model.py --timeout 18000 -k full ; fail+=$?
  LLAMA_DIR=$llama90b WH_ARCH_YAML=$wh_arch_yaml pytest models/tt_transformers/tests/test_model_prefill.py --timeout 1800; fail+=$?

  # Record the end time
  end_time=$(date +%s)
  duration=$((end_time - start_time))
  echo "LOG_METAL: run_t3000_llama3.2-90b-vision_nightly_tests $duration seconds to complete"
  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

run_t3000_tests() {
  # Run Llama3.2-90B Vision tests
  run_t3000_llama3.2-90b-vision_nightly_tests
}

fail=0
main() {
  # For CI pipeline - source func commands but don't execute tests if not invoked directly
  if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "Script is being sourced, not executing main function"
    return 0
  fi

  if [[ -z "$TT_METAL_HOME" ]]; then
    echo "Must provide TT_METAL_HOME in environment" 1>&2
    exit 1
  fi

  if [[ -z "$ARCH_NAME" ]]; then
    echo "Must provide ARCH_NAME in environment" 1>&2
    exit 1
  fi

  # Run all tests
  cd $TT_METAL_HOME
  export PYTHONPATH=$TT_METAL_HOME

  run_t3000_tests

  if [[ $fail -ne 0 ]]; then
    exit 1
  fi
}

main "$@"
