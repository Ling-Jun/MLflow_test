�	� �m��@� �m��@!� �m��@	pT	PA�?pT	PA�?!pT	PA�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$� �m��@-$`ty�?A*�TE@Y�f*�#�?*	�Zd;�Q@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatR�(�1�?!�h^��SB@)'��b�?1���֙E?@:Preprocessing2U
Iterator::Model::ParallelMapV2�аu��?!�E��6!4@)�аu��?1�E��6!4@:Preprocessing2F
Iterator::ModelJ���?!���*�xC@)P÷�n��?1�a/q�2@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�E�~��?!��(�E�2@)CX�%���?1����'@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceMۿ�Ҥt?!7͹�1@)Mۿ�Ҥt?17͹�1@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipY��#��?!S,$�X�N@)����*4p?1}?�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorX9��v�o?!}ן�@)X9��v�o?1}ן�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9pT	PA�?IW�_}��X@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	-$`ty�?-$`ty�?!-$`ty�?      ��!       "      ��!       *      ��!       2	*�TE@*�TE@!*�TE@:      ��!       B      ��!       J	�f*�#�?�f*�#�?!�f*�#�?R      ��!       Z	�f*�#�?�f*�#�?!�f*�#�?b      ��!       JCPU_ONLYYpT	PA�?b qW�_}��X@Y      Y@q_^-���?"�
device�Your program is NOT input-bound because only 0.8% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 