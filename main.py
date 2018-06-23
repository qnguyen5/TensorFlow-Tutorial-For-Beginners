from load_data import *
from softmax_nn import *
import random

num_experiment = 3

for experiment in range(num_experiment):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())  #tf.initialize_all_variables() is deprecated
        max_iteration = 201
        for i in range(max_iteration):
            _, loss_value = sess.run([train_op, loss], feed_dict = {x: images28, y: labels})
            if i % 20 == 0: #print out loss value every 10 session, range 201 allow to get the last loss value of the 200 iteration
                print("i = ", i, "loss_value:", loss_value)

        print("TRAINING FINISHED!\n")


        #PRELIMINARY ASSESSMENT OF MODEL GOODNESS
        """
        # Pick 10 random images
        sample_indexes = random.sample(range(len(images28)), 10)
        sample_images = [images28[i] for i in sample_indexes]
        sample_labels = [labels[i] for i in sample_indexes]

        # Run the "correct_pred" operation
        predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]

        # Print the real and predicted labels
        print(sample_labels)
        print(predicted)

        # Display the predictions and the ground truth visually.
        fig = plt.figure(figsize=(10, 10))
        for i in range(len(sample_images)):
            truth = sample_labels[i]
            prediction = predicted[i]
            plt.subplot(5, 2,1+i)
            plt.axis('off')
            color='green' if truth == prediction else 'red'
            plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction),
                     fontsize=12, color=color)
            plt.imshow(sample_images[i],  cmap="gray")

        plt.show()
        """

        # RUN PREDICTIONS AGAINST FULL TRAINING SET
        predicted = sess.run([correct_pred], feed_dict={x: images28})[0]
        # Calculate correct matches
        match_count = sum([int(y == y_) for y, y_ in zip(list(labels), predicted)])
        # Calculate the accuracy
        accuracy = match_count/labels.size
        # Print the accuracy
        print("ACCURACY ON TRAIN SET: {:.3f}".format(accuracy))

        # RUN PREDICTIONS AGAINST FULL TEST SET
        predicted = sess.run([correct_pred], feed_dict={x: test_images28})[0]
        # Calculate correct matches
        match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])
        # Calculate the accuracy
        accuracy = match_count / len(test_labels)
        # Print the accuracy
        print("ACCURACY ON TEST SET: {:.3f}".format(accuracy))

        print("EXPERIMENT {0} DONE".format(experiment))
        print("-----------------------------------------------------")
