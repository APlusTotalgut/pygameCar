class PerformanceMeasurement:
    def __init__(self):
        self.packages = [[0,0]] # gets added a new list every package_refresh
        self.packageNumber = 0
        self.episode_progress = 0
        self.package_refresh = 100000
        self.starting_episode = 0

    def addTurn(self, current_episode, success = True):
        intern_current_episode = (self.episode_progress + self.starting_episode)
        if current_episode != (intern_current_episode + 1):

            aproximate_failed_episodes = current_episode - intern_current_episode - 1

            self.packages[self.packageNumber][0] += aproximate_failed_episodes

        if success:
            self.episode_progress = current_episode - self.starting_episode
            self.packages[self.packageNumber][1] += 1
        
        if self.packages[self.packageNumber][0] + self.packages[self.packageNumber][1] > self.package_refresh:
            self.packages.append([0,0])
            self.packageNumber += 1

    def getAccuracy(self):
        if self.packages[self.packageNumber][1] == 0:
            return 0
            
        return str((self.packages[self.packageNumber][1] / (self.packages[self.packageNumber][0] + self.packages[self.packageNumber][1])) * 100) + "%"
        #return str(self.packages[self.packageNumber][1]) + ' ' + str(self.packages[self.packageNumber][0])
