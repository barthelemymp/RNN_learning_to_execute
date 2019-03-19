

def trainIters(encoder, decoder, n_iters,epochs=5, print_every=100, plot_every=100, learning_rate=0.01,n_evaluate=1000):
 
    all_losses=[]
    all_test_losses=[]

    start = time.time()
    print_loss_total = 0.  # Reset every print_every
    plot_loss_total = 0. # Reset every plot_every


    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [random.choice(pairs)for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for epoch in range(0,epochs):

        for iter in range(1, n_iters + 1):
            training_pair = training_pairs[iter - 1]
            input_tensor = torch.tensor(training_pair[0])
            target_tensor = torch.tensor(training_pair[1])

            loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print("epoch :" + str(epoch) + " iter : " + str(iter))
                print( print_loss_avg)


            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_loss_total = 0.
                all_losses+=[plot_loss_avg]

                test_loss=0.
                for i in range(n_evaluate):
                    test_loss+=evaluate(test_pair=test_pairs[i],encoder=encoder, decoder=decoder, criterion=criterion, max_length=MAX_LENGTH)
                all_test_losses+=[test_loss/n_evaluate]


    torch.save(encoder,"saved_model/encoder")
    torch.save(encoder,"saved_model/second_decoder")       
    np.save('result/all_losses',np.array(all_losses))
    np.save('result/all_test_losses',np.array(all_losses))