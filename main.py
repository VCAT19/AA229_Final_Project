import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import pandas as pd

# Import dataset
df = pd.read_csv('20_21_NBA_Head_to_Head.csv', index_col=1)

# Problem Set up
n = 10  # Trials
Team_1 = 'ATL'
Team_2 = 'Chicago Bulls'
Reward_predict_correct = 1
Reward_predict_wrong = 0


def Multi_Arm_Bandit_NBA_1(n, Team_1, Team_2):
    if Team_1 == Team_2:
        print('*************************WARNING*************************')
        print('You designated "Team_1" and "Team_2" as the same NBA team.')
        print('A team does not play itself in an official NBA game.')
        print('Please select two unique teams as "Team 1" and "Team 2."')
        print('*********************************************************')
        return 1

    # Hardcoded for now to Align to ATL record against CHI, will un-junk later
    theta = df.iat[0, 4]
    beta_parts = theta.split('-')

    # Create Prior based off NBA Team Head-to-Head Record
    # beta_a = team win, beta_b = team lose
    beta_a = float(beta_parts[0])
    beta_b = float(beta_parts[1])

    # Cheat if there is a 0 in the data set as beta distributions don't take 0's.
    # We don't want to give a team a "free win" in the pseudo counts before adding on actual 20-21 season results.
    if int(beta_b) == 0:
        beta_b = .0001
    if int(beta_a) == 0:
        beta_a = .0001

    # Simulate Predicting the outcome of a game.
    Q_predict_team_1_win = float(0)  # Start with policies of 0
    Q_predict_team_1_loss = float(0)
    a = 'win' # Predict that Team_1 wins first because who likes losing!
    # Keep track of how well the model predicts
    Wrong_prediction = 0
    Correct_prediction = 0
    # Keep track of how many times a prediction of win "wins"/i.e. is correct and how many times a prediction of loss "losses"/i.e. is incorrect
    win_wins = 0
    win_losses = 0
    # Keep track of how many times a prediction of loss "wins"/i.e. is correct and how many times a prediction of loss "losses"/i.e. is incorrect
    loss_wins = 0
    loss_losses = 0
    # Keep track of game wins and losses
    wins = 2
    losses = 2
    for i in range(n):
        #r = beta.rvs(beta_a, beta_b, size=1)
        r = beta.rvs(wins, losses, size=1)

        # Update Psuedo Counts
        if r >= .5:
            wins += 1
        else:
            losses += 1

        if ((r >= .5) and (a == 'win')):
            win_wins += 1
            Correct_prediction += 1
            Q_new = ((wins + 1) / (win_losses + win_wins + 2)) * (1 + Q_predict_team_1_win) + (1 - (win_wins + 1) / (win_losses + win_wins + 2)) * (0 + Q_predict_team_1_win)
            Q_predict_team_1_win = Q_new

        elif ((r <= .5) and (a == 'win')):
            win_losses += 1
            Wrong_prediction += 1
            Q_new = ((wins + 1) / (win_losses + win_wins + 2)) * (1 + Q_predict_team_1_win) + (1 - (win_wins + 1) / (win_losses + win_wins + 2)) * (0 + Q_predict_team_1_win)
            Q_predict_team_1_win = Q_new


        elif((r <= .5) and (a == 'loss')):
            loss_wins += 1
            Correct_prediction += 1
            Q_new = ((loss_wins + 1) / (loss_losses + loss_wins + 2)) * (0 + Q_predict_team_1_loss) + (1 - (loss_wins + 1) / (loss_losses + loss_wins + 2)) * (1 + Q_predict_team_1_loss)
            Q_predict_team_1_loss = Q_new

        elif ((r >= .5) and (a == 'loss')):
            loss_losses += 1
            Wrong_prediction += 1
            Q_new = ((loss_wins + 1) / (loss_losses + loss_wins + 2)) * (0 + Q_predict_team_1_loss) + (
                        1 - (loss_wins + 1) / (loss_losses + loss_wins + 2)) * (1 + Q_predict_team_1_loss)
            Q_predict_team_1_loss = Q_new

        print(i, r, a, wins, losses, win_wins, win_losses, loss_wins, loss_losses)

        # Update Policy based on prediction results
        if Q_predict_team_1_win > Q_predict_team_1_loss:
            a == 'win'
        else:
            a == 'loss'

    print('Precentage Correct:' + str(Correct_prediction/n))
    print('Precentage Incorrect:' + str(Wrong_prediction/n))
    print(Q_predict_team_1_win)
    print(Q_predict_team_1_loss)


if __name__ == '__main__':
    Multi_Arm_Bandit_NBA_1(n, Team_1, Team_2)
