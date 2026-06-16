# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for SandboxCliAgent pure helpers."""

from nemo_gym.sandbox_cli_agent import swebench_image_tag


def test_swebench_image_tag_rewrites_double_underscore_and_lowercases():
    assert swebench_image_tag("astropy__astropy-12907") == "astropy_1776_astropy-12907"
    # the org segment is lower-cased too
    assert swebench_image_tag("PyCQA__flake8-1234") == "pycqa_1776_flake8-1234"


def test_swebench_image_tag_leaves_non_swebench_ids_untouched():
    # no "__" => not a SWE-bench id => pass through unchanged (preserve case)
    assert swebench_image_tag("my-Custom-Image_42") == "my-Custom-Image_42"
