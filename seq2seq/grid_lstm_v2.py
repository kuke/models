#!/usr/bin/env python
#coding:gbk

import math
import sys
import paddle.v2 as paddle

#source_language_dict_dim = 19554  #english
#target_language_dict_dim = 21787  # ch_basic

source_language_dict_dim = 30000  #english
target_language_dict_dim = 30000  # ch_basic

latent_chain_dim = 256  #1024
word_vec_dim = latent_chain_dim * 2  #1024 #620

eos_id = 1
start_id = 0
generating = False


def split_layer(name, inputs, size):
    with paddle.layer.mixed(name=name + "_first_half", size=size) as first_half:
        first_half += paddle.layer.identity_projection(input=inputs, offset=0)
    with paddle.layer.mixed(name=name + "_last_half", size=size) as last_half:
        last_half += paddle.layer.identity_projection(input=inputs, offset=size)
    return first_half, last_half


def grid_lstm_net(source_dict_dim, target_dict_dim, generating=False):
    ###############################lstm unit
    def lstm_recurrent_group(name,
                             size,
                             active_type,
                             state_active_type,
                             gate_active_type,
                             inputs,
                             parameter_name=None,
                             boot_layer=None,
                             state_boot_layer=None,
                             seq_reversed=False):
        input_all_dimensions_layer_name = name + "_" + "input_all_dimensions"

        global out_memory, state_memory, input_all_dimensions_layer

        def lstm_recurrent_step(inputs):
            global out_memory, state_memory, input_all_dimensions_layer
            out_memory = paddle.layer.memory(
                name=name, size=size, boot_layer=boot_layer)
            state_memory = paddle.layer.memory(
                name=name + "_" + "state",
                size=size,
                boot_layer=state_boot_layer)
            input_all_dimensions_layer = paddle.layer.concat(
                name=input_all_dimensions_layer_name,
                input=[out_memory, inputs])
            with paddle.layer.mixed(size=size * 4) as lstm_inputs:
                lstm_inputs += paddle.layer.full_matrix_projection(
                    input=input_all_dimensions_layer)

            lstm_step = paddle.layer.lstm_step(
                name=name,
                size=size,
                act=active_type,
                state_act=state_active_type,
                gate_act=gate_active_type,
                inputs=lstm_inputs,
                state=state_memory)
            state_memory_out = paddle.layer.get_output(
                name=name + "_" + "state", input=lstm_step, arg_name='state')
            return lstm_step

        input_concat_layer = paddle.layer.concat(input=inputs)
        group_inputs = [
            paddle.layer.StaticInputV2(input=input_concat_layer, is_seq=True)
        ]

        group_outs = paddle.layer.recurrent_group(
            name="lstm_decoder_group",
            input=group_inputs,
            step=lstm_recurrent_step, )
        return input_all_dimensions_layer, out_memory, state_memory

    ##################### network ###############################
    src_word_id = paddle.layer.data(
        name='source_language_word',
        type=paddle.data_type.integer_value_sequence(source_dict_dim))

    trg_word_id = paddle.layer.data(
        name='target_language_word',
        type=paddle.data_type.integer_value_sequence(target_dict_dim))
    trg_next_word_id = paddle.layer.data(
        name='target_language_next_word',
        type=paddle.data_type.integer_value_sequence(target_dict_dim))

    ############################### encoder ###########################
    src_embedding = paddle.layer.embedding(
        name="source_embedding",
        input=src_word_id,
        size=word_vec_dim,
        param_attr=paddle.attr.ParamAttr(name='_source_language_embedding'))
    src_embedding_first_half, src_embedding_last_half = split_layer(
        "source_embedding", src_embedding, latent_chain_dim)

    def grid_lstm_step(trg_embedding_inputs):
        ############################### deocder #########################
        memory_decoder_lstm1_out = paddle.layer.memory(
            name="decoder_lstm1",
            size=latent_chain_dim,
            boot_layer=src_embedding_first_half,
            is_seq=True)
        memory_decoder_lstm2_out = paddle.layer.memory(
            name="decoder_lstm2",
            size=latent_chain_dim,
            boot_layer=src_embedding_last_half,
            is_seq=True)

        #########################grid lstm start
        trg_embedding_first_half, trg_embedding_last_half = split_layer(
            "target_embedding", trg_embedding_inputs, latent_chain_dim)

        all_dimensions_input_lstm1, anotation_lstm1, anotation_lstm1_state = lstm_recurrent_group(
            name="anotation_lstm1",
            size=latent_chain_dim,
            active_type=paddle.activation.Tanh(),
            state_active_type=paddle.activation.Tanh(),
            gate_active_type=paddle.activation.Sigmoid(),
            inputs=[memory_decoder_lstm1_out],
            parameter_name="anotation_lstm1.w",
            boot_layer=trg_embedding_first_half,
            state_boot_layer=trg_embedding_last_half, )

        with paddle.layer.mixed(size=4 * latent_chain_dim) as lstm1_input:
            lstm1_input += paddle.layer.full_matrix_projection(
                input=all_dimensions_input_lstm1,
                param_attr=paddle.attr.ParamAttr(name="decoder_lstm1.w"))
        # lstm1
        decoder_lstm1_state = paddle.layer.memory(
            name="decoder_lstm1_state",
            size=latent_chain_dim,
            boot_layer=src_embedding_last_half)
        decoder_lstm1 = paddle.layer.lstm_step(
            name="decoder_lstm1",
            size=latent_chain_dim,
            act=paddle.activation.Tanh(),
            state_act=paddle.activation.Tanh(),
            gate_act=paddle.activation.Sigmoid(),
            input=lstm1_input,
            state=decoder_lstm1_state)
        decoder_lstm1_state_out = paddle.layer.get_output(
            name="decoder_lstm1_state", input=decoder_lstm1, arg_name='state')

        with paddle.layer.mixed(
                name="grid_layer1_out",
                size=latent_chain_dim) as grid_layer1_out:
            grid_layer1_out += paddle.layer.full_matrix_projection(
                input=anotation_lstm1)
            grid_layer1_out += paddle.layer.full_matrix_projection(
                input=decoder_lstm1)

        anotation_lstm1_last = paddle.layer.last_seq(input=anotation_lstm1)
        anotation_lstm1_state_last = paddle.layer.last_seq(
            input=anotation_lstm1_state)

        all_dimensions_input_lstm2, anotation_lstm2, anotation_lstm2_state = lstm_recurrent_group(
            name="anotation_lstm2",
            size=latent_chain_dim,
            active_type=paddle.activation.Tanh(),
            state_active_type=paddle.activation.Tanh(),
            gate_active_type=paddle.activation.Sigmoid(),
            inputs=[memory_decoder_lstm2_out, grid_layer1_out],
            parameter_name="anotation_lstm2.w",
            seq_reversed=True,
            boot_layer=anotation_lstm1_last,
            state_boot_layer=anotation_lstm1_state_last, )

        with paddle.layer.mixed(size=4 * latent_chain_dim) as lstm2_input:
            lstm2_input += paddle.layer.full_matrix_projection(
                input=all_dimensions_input_lstm2,
                param_attr=paddle.attr.ParamAttr(name="decoder_lstm2.w"))

        decoder_lstm2_state = paddle.layer.memory(
            name='decoder_lstm2_state',
            size=latent_chain_dim,
            boot_layer=src_embedding_last_half,
            is_seq=True)
        decoder_lstm2 = paddle.layer.lstm_step(
            name="decoder_lstm2",
            size=latent_chain_dim,
            act=paddle.activation.Tanh(),
            state_act=paddle.activation.Tanh(),
            gate_act=paddle.activation.Sigmoid(),
            inputs=lstm2_input,
            state=decoder_lstm2_state)
        decoder_lstm2_state_out = paddle.layer.get_output(
            name='decoder_lstm2_state', input=decoder_lstm2, arg_name='state')

        decoder_layer2_out = paddle.layer.concat(
            input=[anotation_lstm2, anotation_lstm2_state])
        decoder_out = paddle.layer.first_seq(input=decoder_layer2_out)
        with paddle.layer.mixed(
                size=target_language_dict_dim,
                act=paddle.activation.Softmax(),
                bias_attr=paddle.attr.ParamAttr(name="_output.b")) as output:
            output += paddle.layer.full_matrix_projection(
                input=decoder_out,
                param_attr=paddle.attr.ParamAttr(name="_output.w"))
        return output

    #########################grid lstm end
    decoder_group_name = 'grid_decoder_group'
    if  generating:
        '''
        predict_word_memory = paddle.layer.memory(
                name = "predict_word",
                size = target_dict_dim,
                boot_with_const_id = start_id,)
        predict_embedding = paddle.layer.embedding(
                name = "target_embedding",
                input=predict_word_memory,
                size = word_vec_dim,
                param_attr = paddle.attr.ParamAttr(name = '_target_language_embedding"'))
        trg_embedding = paddle.layer.GeneratedInputV2(
            size=target_dict_dim,
            embedding_name='_target_language_embedding',
            embedding_size=word_vector_dim)
        group_inputs = []
        group_inputs.append(trg_embedding)

        beam_gen = paddle.layer.max_search(
            name=decoder_group_name,
            step=lstm_recurrent_step,
            input=group_inputs,
            bos_id=0,
            eos_id=1,
            beam_size=beam_size,
            max_length=max_length)
        return beam_search
        '''
    else:

        trg_embedding = paddle.layer.embedding(
            input=trg_word_id,
            size=word_vec_dim,
            param_attr=paddle.attr.ParamAttr(name='_target_language_embedding'))
        group_inputs = [
            paddle.layer.StaticInputV2(input=trg_embedding, is_seq=True)
        ]

        decoder_output = paddle.layer.recurrent_group(
            name=decoder_group_name, input=group_inputs, step=grid_lstm_step)
        label = paddle.layer.data(
            name='target_language_next_word',
            type=paddle.data_type.integer_value_sequence(target_dict_dim))
        cost = paddle.layer.classification_cost(
            input=decoder_output, label=label)
        return cost


def train():
    cost = grid_lstm_net(source_language_dict_dim, target_language_dict_dim)
    parameters = paddle.parameters.create(cost)
    # optimizer
    optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3,
        gradient_clipping_threshold=10.0, )

    #trainer 
    trainer = paddle.trainer.SGD(
        cost=cost, parameters=parameters, update_equation=optimizer)
    #data reader 
    wmt14_reader = paddle.batch(
        paddle.reader.shuffle(
            paddle.dataset.wmt14.train(source_dict_dim), buf_size=8192),
        batch_size=22)

    def event_handler(event):
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 100 == 0 and event.batch_id > 0:
                with gzip.open('params_grid_lstm_batch_%d.tar.gz' %
                               event.batch_id, 'w') as f:
                    parameters.to_tar(f)

            if event.batch_id % 10 == 0:
                print "\n Pass %d, Batch %d, Cost %d " % (
                    event.pass_id, event.batch_id, event.cost, event.metrics)
            else:
                sys.stdout.write('.')
                sys.stdout.flush()

    trainer.train(reader=wmt14_reader, event_handler=event_handler, num_pass=2)


def generate():
    gen_creator = paddle.dataset.wmt14.gen(source_dict_dim)
    gen_data = []
    gen_num = 3
    for item in gen_creator():
        gen_data.append((item[0], ))
        if len(gen_data) == gen_num:
            break
    beam_gen = grid_lstm_net(source_language_dict_dim, target_language_dict_dim,
                             True)
    #get model
    parameters = paddle.dataset.wmt14.model()
    beam_result = paddle.infer(
        output_layer=beam_gen,
        parameters=parameters,
        input=gen_data,
        field=['prob', 'id'])

    #get th dictionary
    src_dict, trg_dict = paddle.dataset.wmt14.get_dict(source_dict_dim)

    seq_list = []
    seq = []
    for w in beam_result[1]:
        if w != -1:
            seq.append(w)
        else:
            seq_list.append(' '.join([trg_dict.get(w) for w in seq[1:]]))
            seq = []

    prob = beam_result[0]
    beam_size = 3
    for i in xrange(gen_num):
        print "\n*******************************"
        print "sec:", ' '.join([src_dict.get(w) for w in gen_data[i][0]]), "\n"
        for j in xrange(beam_size):
            print "prob = %f:" % (prob[i][j]), seq_list[i * beam_size + j]


def main():
    paddle.init(use_gpu=False, trainer_count=4)

    if generating:
        generate()
    else:
        train()


if __name__ == '__main__':
    main()
