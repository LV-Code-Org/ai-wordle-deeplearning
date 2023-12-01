from random import choice

# initialize answer list and possible guess list (valid words)
ANSWERS = []

with open("txt/answers.txt") as file:
    lines = [line.rstrip() for line in file]
    ANSWERS = lines[0].split(" ")

def load_wordlist() -> list:
    with open("txt/list.txt") as file:
        lines = [line.rstrip() for line in file]
        return lines

WORD_LIST = load_wordlist()


def guessRandom() -> str:
    """Returns a random 5-letter word."""
    return choice(WORD_LIST)


class WordleAI:

    def __init__(self) -> None:
        self.word = choice(ANSWERS)
        self.guesses = []

    def reset(self) -> None:
        self.word = choice(ANSWERS)
        self.guesses = []

    def play_step(self, action: str) -> list:
        """Action should be the guess."""

        # check if game over
        game_over = False
        reward = 0
        results = self._score(action)
        self.guesses.append(action)
        print(f"ACTION: {action} | ANSWER: {self.word} | RESULTS: {results}")

        if results[0] == 5:  # all green
            game_over = True
            reward = 10 - len(self.guesses)
            return [reward, game_over, len(self.guesses)]

        return [reward, game_over, len(self.guesses)]

    def _score(self, guess: str) -> list[int]:
        """Returns [green, yellow, gray] for a given guess."""
        answerList = list(self.word)
        guessList = list(guess)

        green = sum(1 for a, g in zip(answerList, guessList) if a == g)
        yellow = sum(1 for g in guessList if g in answerList and g !=
                     answerList[guessList.index(g)])
        gray = len(guessList) - green - yellow

        return [green, yellow, gray]


    def get_state(self) -> list[int]:
        if not self.guesses:
            return None  # Return None to indicate an invalid state

        # Ensure there is at least one guess before attempting to calculate the state
        state = []
        for letter in self.guesses[-1]:
            if letter in self.word:
                if letter == self.word[self.guesses[-1].index(letter)]:
                    state.append(1)  # green
                else:
                    state.append(2)  # yellow
            else:
                state.append(3)  # gray

        # Padding to make the state length 5
        state += [0] * (5 - len(state))

        return state
