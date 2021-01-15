import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns; sns.set()
from sklearn.cluster import KMeans
import time
from tqdm import tqdm
import sys


start_time_global = time.time()

def scatter_generator(N):
    np.random.seed(19680801)
    i=0
    coords=[[np.random.rand(),np.random.rand()]]
    x = np.random.rand()
    y = np.random.rand()
    x_coord=[coords[0][0]]
    y_coord=[coords[0][1]]
    individual_coord=[x,y]
    while i <= N:
        x = np.random.rand()
        y = np.random.rand()
        individual_coord=[x,y]
        coords.append(individual_coord)
        x_coord.append(coords[i][0])
        y_coord.append(coords[i][1])
        i+=1
    coords=np.array(coords)
    x_coord.pop(0)
    y_coord.pop(0)
    return coords
max_k=100
puntos=[]
tiempo=[]
data=[[0,0,0]]
l=1
max_points=100000
with tqdm(total=100, file=sys.stdout) as pbar:
    k=1
    while k <=max_k:
        n_points = 100
        while n_points<=max_points:
            start_time = time.time()
            kmeans = KMeans(n_clusters=k, random_state=0).fit(scatter_generator(n_points))
            #print(k)
            #print(kmeans.labels_)
            #kmeans.predict([[0, 0], [12, 3]])
            kmeans.cluster_centers_
            end_time=(time.time() - start_time)
            #print(end_time)
            puntos.append(n_points)
            tiempo.append(end_time)
            n_points=n_points+500
            current_value=[k,n_points,end_time]
            data.append(current_value)
        #plt.scatter(puntos, tiempo, alpha=0.5)
        #plt.show()
        pbar.update(l)
        k=k+1
data.pop(0)
df = pd.DataFrame(data, columns = ['k', 'N_Points','Time'])

fig = pyplot.figure()
ax = Axes3D(fig)
ax.set_xlabel('k')
ax.set_ylabel('N_Points')
ax.set_zlabel('Time [min]')
ax.scatter(df.k.values.tolist(), df.N_Points.values.tolist(), df.Time.values.tolist())
pyplot.show()
df.to_csv('K_Means_Performance_Results.csv')
#plt.savefig('3d Results.png')
end_time_global=(time.time() - start_time_global)
end_time_min=end_time_global/60
text=["Max points: "+str(max_points)+"Max Clusters: "+str(max_k)+"Time to complete code: "+str(end_time_global)+" seconds, or: "+str(end_time_min)+" minutes.\n"]
f= open("Code_out_log_KMEANS_PERFORMANCE.txt","w+")
f.writelines(text)
f.close()
