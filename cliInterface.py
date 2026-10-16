from prompt_toolkit import PromptSession
import sys

class CliInterface:
    
    def __init__(self, performanceMeasurement, neuronalNetwork):
        self.performanceMeasurement = performanceMeasurement
        self.neuronal = neuronalNetwork
        self.session = PromptSession(
            bottom_toolbar=self.toolbar,
            refresh_interval=0.4
        )

    def updateLine(line):
        sys.stdout.write("\r\033[2K" + line)
        sys.stdout.flush()

    def display(self):
        user_input = ''
        while user_input != 'exit':
            user_input = self.session.prompt(">> ")

    def toolbar(self):
        return f"Episode {self.neuronal.episode} | Accuracy {self.performanceMeasurement.getAccuracy()}"

