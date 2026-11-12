from autograd import grad
from colorama import Fore, Style
import math
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
    def formatList(d):
        if not isinstance(d, list):
            d = list(d)
        return d


    @staticmethod
    def mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)


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


    @staticmethod
    def roundUp(v, m=5):
        """
        :param v: round this value
        :param m: round value to a multiple of m
        """
        return int(math.ceil(v / m) * m)


    def getAxisTicks(self, values, largeValues=True, customSteps=None,
                     addZero=True):
        values, padding, roundBy  = self.formatList(values), 0.05, 5
        maxVal, padAdj  = int(max(values)), 10
        if customSteps is None:
            customSteps = []
        if customSteps:
            steps = customSteps
        else:
            if largeValues:
                steps = [25, 20, 30, 40, 50, 25, 15, 45, 75, 10, 5]
            else:
                steps = [4, 3, 2, 1]
                padding = 0.05
        dist = self.roundUp(maxVal - values[0], m=roundBy)
        ticks = []
        for step in steps:
            if dist % step == 0:
                ticks = list(np.arange(values[0], maxVal + step, step))
                if len(ticks) > 3:
                    padAdj = step
                    break
        if ticks and addZero and 0 not in ticks:
            ticks = np.insert(ticks, np.searchsorted(ticks, 0), 0)
        # print(f'Axis ticks: {ticks}\n')
        padding = int(padding * dist/2)
        padding = padAdj / 10 if padding == 0 else padding
        
        return list(ticks), padding

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


    def forwardEuler(self, fPrime, u0, time, nDatapoints:list,
                     pHeader=True):
        """
        A numerical solution to an ODE
        :param fPrime: derivative of a function, format: f(t, u)
        :param u0: initial value
        :param time: [int or float] duration, or maximum value of the x-axis
        :param nDatapoints: [list] Number of datapoint for a curve
        :param pHeader: [bool] print bar, [str] print bar with label
        :return: [list] pd.Dataframes of each solution found at different dt values
        """
        if pHeader:
            if isinstance(pHeader, str):
                self.header(txt=pHeader)
            else:
                self.header()

        datasets = []
        for N in nDatapoints:
            dt = time / N
            t = np.zeros(N+1)
            u = np.zeros(N+1)
            print(f'Timestep: {red}Δt={dt}{rst}')

            # Evaluate
            u[0] = u0
            for n in range(N):
                t[n + 1] = t[n] + dt
                u[n + 1] = u[n] + dt * fPrime(t[n], u[n])
            d = pd.DataFrame(u, index=t, columns=[f'Euler, Δt={dt}'])
            datasets.append(d)
            print(f'{d}\n\n')

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
        """
        :param data: [pd.DataFrame or list of pd.DataFrames]
        :param title: [str] figure title
        :param colors: [list] colors for the lines
        :param figName: [str] file name for the saved figure
        :param labelX: [str] axis label
        :param labelY: [str] axis label
        :param yRotate: [int] rotate the y-label
        :param figDir: [str] save the fig at self.figDir/{figDir}/
            if an empty string is provided the fig will be saved at self.figDir/
        :param addGrid: [bool] add grid to the figure
        :param saveFig: [bool] True: saves the fig, False: does not save the fig
        """

        # Verify that there are a sufficient number of strings
        colors = self.formatList(colors)
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
        xTicks = []
        if df:
            xTicks = list(data.index)
            for i in range(numCurves):
                ax.plot(data.index, data.iloc[:,i], color=colors[i],
                    label=data.columns[i], linewidth=self.lineThickness)
        else:
            for i, d in enumerate(data):
                if not xTicks:
                    xTicks = list(d.index)
                for c in range(len(d.columns)):
                    ax.plot(d.index, d.iloc[:, c], color=colors[i],
                            label=d.columns[c], linewidth=self.lineThickness)
        ax.legend(fontsize=self.labelSizeTicks, loc='best', edgecolor='black',
                  framealpha=0.8)
        ax.axhline(y=0, color='black', linewidth=self.lineThickness)

        # Axis params
        ax.set_title(title, fontsize=self.labelSizeTitle, fontweight='bold')
        ax.set_xlabel(labelX, fontsize=self.labelSizeAxis)
        ax.set_ylabel(labelY, fontsize=self.labelSizeAxis, labelpad=20, rotation=yRotate)
        ax.tick_params(labelsize=12, width=self.lineThickness)

        # Set: y-ticks
        if xTicks:
            large = True if max(xTicks) >= 10 else False
            xTicks, pad = self.getAxisTicks(values=xTicks, largeValues=large)
            print(f'Ticks: {xTicks}\n* Pad: {pad}\n')
            if 0 not in xTicks:
                xTicks = np.insert(xTicks, np.searchsorted(xTicks, 0), 0)
            ax.set_xlim(xTicks[0] - pad, xTicks[-1] + pad)
            ax.set_xticks(xTicks)
        elif not xTicks or len(xTicks) > 12:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Set: y-ticks
        yTicks = ax.get_yticks()
        if len(yTicks) > 0:
            large = True if max(yTicks) >= 10 else False
            yTicks, _ = self.getAxisTicks(values=yTicks, largeValues=large)
            if len(yTicks) > 0:
                ax.set_ylim(yTicks[0], yTicks[-1])
                ax.set_yticks(yTicks)
        if len(yTicks) < 1 or len(yTicks) > 12:
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        # Grid
        if addGrid:
            ax.grid(True, linewidth=self.lineThickness, color='lightgrey')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(self.lineThickness)

        plt.tight_layout()
        if saveFig:
            self.saveFig(plot=plt, figureName=figName, pathDir=figDir)
        self.plotFig(
            plot=plt, figureName=figName, pathDir=figDir, fig=fig
        )
