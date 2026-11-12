from autograd import grad
from colorama import Fore, Style
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import os
import pandas as pd
import random
import subprocess
import sys


# Define colors
red = Fore.LIGHTRED_EX
gold = Fore.YELLOW
yellow = Fore.LIGHTYELLOW_EX
green = Fore.GREEN
cyan = Fore.LIGHTCYAN_EX
blue = Fore.BLUE
pink = Fore.LIGHTMAGENTA_EX
magenta = Fore.MAGENTA
white = Fore.LIGHTWHITE_EX
bold = Style.BRIGHT
rst = Style.RESET_ALL


def pressKey(event):
    if event.key == 'escape':
        plt.close()
    elif event.key == 'e':
        sys.exit()
    elif event.key == 'r':
        python = sys.executable
        os.execl(python, python, *sys.argv)


def getAxisTicks(values, largeValues=True, customSteps=None):
    if customSteps is None:
        customSteps = []
    padding = 0.05
    if not isinstance(values, list):
        values = list(values)
    if customSteps is not None:
        steps = customSteps
    else:
        if largeValues:
            steps = [25, 20, 30, 40, 50]
        else:
            steps = [5, 2, 3, 4]
            padding = 0.01

    dist = values[-1] - values[0]
    padding = int(padding * dist)
    ticks = []
    for step in steps:
        if dist % step == 0:
            ticks = list(np.arange(values[0], values[-1] + step, step))
            print(f'Step: {step}')
            break
    # print(f'xTicks: {ticks}\n')
    return ticks, padding


class ODE:
    def __init__(self):
        self.printN = 10
        self.roundVal = 3
        self.headerLen = 80
        self.rstFloat = True
        self.webBrowserFigs = True
        self.overwriteFigs = False

        # Params: Figures
        self.figDir = 'Figures'
        self.figSize = (9.5, 8) # (width, height)
        self.figSizeWide = (9.5, 5)
        self.labelSizeTitle = 20 # Set font-sizeself.figDir='Figures'
        self.labelSizeAxis = 16 # Set font-size
        self.labelSizeTicks = 12 # Set font-size
        self.lineThickness = 1.5
        self.tickLength = 4
        self.figureResolution = 600
        
        self.mkdir(self.figDir)

        # Set options
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', 10)
        pd.set_option('display.width', 1000)
        self.formatFloat()


    def formatFloat(self, sciNotation=False):
        if sciNotation:
            pd.options.display.float_format = lambda x: f'{x:,.{self.roundVal}e}'
            self.rstFloat = True
        elif self.rstFloat:
            pd.options.display.float_format = lambda x: f'{x:,.{self.roundVal}f}'
            self.rstFloat = False


    @staticmethod
    def mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)


    @staticmethod
    def formatList(x):
        if not isinstance(x, list):
            x = list(x)
        return x


    @staticmethod
    def derivative(f, idxDiffArg=1):
        """
        :param idxDiffArg:
            Differentiates with respect to the ith argument of f
            For f(t, u)
                idxDiffArg=0 -> df/dt
                idxDiffArg=1 -> df/du
        """
        return grad(f, argnum=idxDiffArg)


    def header(self, txt=''):
        if txt:
            l = len(txt) + 2
            b = int(np.floor((self.headerLen - l) / 2))
            if l % 2 == 0:
                print(f'{"=" * b} {txt} {"=" * b}')
            else:
                print(f'{"=" * b} {txt} {"=" * (b+1)}')
        else:
            print('='*self.headerLen)


    def forwardEuler(self, f, y0, time, nDatapoints:list, plotFig=False,
                     lineColor='green', figTitle='Euler Method', figName='Euler',
                     figDir='', pHeader=False):
        if pHeader:
            if isinstance(pHeader, str):
                self.header(txt=pHeader)
            else:
                self.header()

        # Get the derivative of f with respect to the first (or 0th) param
        # Ex:
            # f(t, u) -> df/dt
        fPrime = self.derivative(f, idxDiffArg=0)

        datasets = []
        for N in nDatapoints:
            dt = time / N
            t = np.zeros(N+1)
            y = np.zeros(N+1)
            print(f'Timestep: {red}Δt={dt}{rst}')

            # Evaluate
            y[0] = y0
            for n in range(N):
                t[n + 1] = t[n] + dt
                y[n + 1] = y[n] + dt * fPrime(t[n])
            d = pd.DataFrame(y, index=t, columns=[f'Euler, Δt={dt}'])
            datasets.append(d)
            print(f'{d}\n\n')

        if plotFig:
            self.plotLines(
                data=datasets, colors=lineColor, title=figTitle, figName=figName,
                labelX='Time', labelY='y(t)', figDir=figDir
            )

        return datasets

    
    @staticmethod
    def rk4(time, f, h, y0):
        print(f'Runge-Kutta')
        d = pd.DataFrame(0.0, index=time, columns=['RK4'])

        def step(t, u):
            k1 = f(t, u)
            k2 = f(t + h/2, u + h/2 * k1)
            k3 = f(t + h / 2, u + h / 2 * k2)
            k4 = f(t + h / 2, u + h * k3)
            return u + h/6 * (k1 + 2*k2 + 2*k3 + k4)

        # Evaluate
        y = y0
        for i, time in enumerate(d.index):
            d.loc[time, 'RK4'] = y
            y = step(t=time, u=y)
        print(f'Data:\n{d}\n')
        return d

    
    def getFigPath(self, figureName, figDir=''):
        if figDir:
            d = os.path.join(self.figDir, figDir)
            path = os.path.join(d, figureName)
            self.mkdir(d)
        else:
            path = os.path.join(self.figDir, figureName)
        if not path.endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.splitext(path)[0] + '.png'
        return path


    def saveFig(self, plot, figureName, pathDir):
        path = self.getFigPath(figureName, pathDir)
        if os.path.exists(path) and not self.overwriteFigs:
            print(f'{yellow}Figure was already saved at path, the figure was not '
                  f'overwritten:\n\t{path}{rst}\n')
        else:
            plot.savefig(path, bbox_inches='tight')
            print(f'Figure was saved at path:\n\t{green}{path}{rst}\n')
        
    
    def plotFig(self, plot, figureName, pathDir='', fig=None):
        if self.webBrowserFigs:
            path = self.getFigPath(figureName, pathDir)
            if os.path.exists(path):
                plot.savefig(path, bbox_inches='tight')
                subprocess.run(['xdg-open', path])  # Open fig in web browser
        else:
            fig.canvas.mpl_connect('key_press_event', pressKey)
            plt.show()


    def plotLines(self, data, colors, title, figName, labelX, labelY, yRotate=0,
                  figDir='', addGrid=True, saveFig=True):
        colors = self.formatList(colors)
        # Verify that there are a sufficient number of strings
        df = True
        if isinstance(data, pd.DataFrame):
            numCurves = len(data.columns)
        else:
            numCurves = len(data)
            df = False
        c = ['black', 'crimson', 'gold', 'darkgreen', 'lightgreen', 'limegreen',
             'teal', 'turquoise', 'royalblue', 'indigo', 'darkviolet', 'gray']
        diff, n = numCurves-len(colors), len(c)
        if diff > n:
            print(1)
            for _ in range(diff):
                colors.append(c[random.randint(0, n-1)])
        else:
            i = 0
            while diff + 1 > len(colors):
                if i > 30:
                    break
                clr = c[random.randint(0, n-1)]
                if clr not in colors:
                    colors.append(clr)
                i += 1
            diff = numCurves - len(colors)
            if diff != 0:
                for _ in range(diff):
                    colors.append(c[random.randint(0, n - 1)])

        # Plot figure
        fig, ax = plt.subplots(figsize=self.figSize)
        if df:
            for i in range(numCurves):
                ax.plot(data.index, data.iloc[:,i], color=colors[i],
                    label=data.columns[i], linewidth=self.lineThickness)
        else:
            for i, d in enumerate(data):
                for c in range(len(d.columns)):
                    ax.plot(d.index, d.iloc[:, c], color=colors[i],
                            label=d.columns[c], linewidth=self.lineThickness)
        ax.legend(fontsize=self.labelSizeTicks, loc='best',
                  edgecolor='black', framealpha=0.8)

        # Styling
        ax.set_title(title, fontsize=self.labelSizeTitle, fontweight='bold')
        ax.set_xlabel(labelX, fontsize=self.labelSizeAxis)
        ax.set_ylabel(labelY, fontsize=self.labelSizeAxis, labelpad=20, rotation=yRotate)
        ax.tick_params(labelsize=12)

        # Axis params
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        if max(ax.get_ylim()) > 1:
            ax.set_ylim(top=int(np.ceil(ax.get_ylim()[1])))

        # Grid
        if addGrid:
            ax.grid(True, linewidth=0.25, color='black')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        for _, spine in ax.spines.items():
            spine.set_visible(True)

        plt.tight_layout()
        if saveFig:
            self.saveFig(plot=plt, figureName=figName, pathDir=figDir)
        self.plotFig(
            plot=plt, figureName=figName, pathDir=figDir, fig=fig
        )
