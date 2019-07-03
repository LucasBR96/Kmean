
import numpy
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.utils
#KMEANS_CLUSTER-------------------------------------------------------------
def kCluster( Data , n , maxIt = 30 , clock = 5 , step = 1):

    #initializing clusters-----------------------------------------------------------------------
    x,y = Data.shape
    centroids = numpy.zeros( shape = ( n , y ) )
    
    pos = numpy.array( range(x) )
    numpy.random.shuffle( pos )
    for i in range(n):
        centroids[i] = Data[ pos[ i ] ]

    for it in range( maxIt ):

        #finding closest clentroid ----------------------------------------------------------------------
        A = numpy.zeros( shape = ( x , n , y ) )
        for i in range( x ):
            A[ i ] = Data[i] - centroids #A[i,j,k] = Data[i,j] - centroids[k,j]
        
        clusters = numpy.argmin( ( A*A ).sum( axis = 2 ) , axis = 1 ) 

        #updating clusters-------------------------------------------------------------------------------
        counts = numpy.zeros( n , dtype = numpy.int32 )
        cMeans = numpy.zeros( shape = ( n , y ) )
        
        for i in range( x ):
            cluster = clusters[i]
            cMeans[ cluster ] += Data[ i ]
            counts[ cluster ] += 1
        
        cMeans = ( cMeans.T/counts ).T  # centroids[ i , j ] /= counts[i]
        centroids += ( cMeans - centroids )*step
        
        #showing costs----------------------------------------------------------------------------------
        if clock != 0:
            if ( it + 1)%clock == 0:
                
                custo = kCost(Data , clusters , centroids)
                print( "custo",( it + 1),"=", custo , sep = " ")
        else:
            print("." , end = '')

    return centroids

def closestCentroids( Data , centroids ):

    x , y = Data.shape
    n = centroids.shape[0]

    A = numpy.zeros( shape = ( x , n , y ) )
    for i in range( x ):
        A[ i ] = Data[i] - centroids #A[i,j,k] = Data[i,j] - centroids[k,j]
    
    return numpy.argmin( ( A*A ).sum( axis = 2 ) , axis = 1 ) 

def clusterInfo( Data , clustersMap ):
    '''
    returns for every label of every cluster the mean and standard deviation
    '''
    x,y = Data.shape
    n = int( numpy.max( clustersMap ) + 1 )

    S = numpy.zeros( ( n , y , 2) )
    for c in range( n ):
    
        pos = numpy.where( clustersMap == c )
        cluster =  Data[ pos ] 

        mu = numpy.mean( cluster , axis = 0)
        sigma = numpy.std( cluster , axis = 0 )

        S[ c , : , 0 ] = mu 
        S[ c , : , 1 ] = sigma
    
    return S

def kCost( Data , clusters , centroids):

    custo = 0
    n = max( clusters ) + 1
    for i in range( n ):
        pos = numpy.where( clusters == i )
        members = Data[ pos ]
        D = ( members - centroids[i] )**2
        custo += D.sum()
    
    return custo/Data.shape[0]

def BestVal( Data , cmax = 10):
    '''
    Guesses the ideal number of clusters for data set
    '''

    print( )
    print("CLUSTERS","CUSTO", "GRAD", sep = "\t")

    costs = [0,0]
    for i in range( 2 , cmax + 1 ):

        centroids = kCluster( Data , i ,clock = 0 )
        clusters = closestCentroids( Data , centroids )
        cost = kCost( Data , clusters , centroids )

        costs.append( cost )
        if i == 2: Grad = "-" 
        else:
            Grad = costs[i - 1] - cost

        print( i , cost , Grad , sep = '\t')

    print( "insira melhor valor" )
    result = int( input() )
    print()
    return result

#LOADINDING DATASETS ------------------------------------------------------------------------------------------------------------------------------------------------------
def loadLA_Car_Accidents():
    horas , coord = [] , []
    with open( 'E:\Datasets\LAaccidentData.csv') as acc:
        acc.readline()
        while True:

            report = acc.readline()
            if not report:
                break 

            report = report.split( sep = ",")
            hor  = report[3].rjust(4 , "0"  )

            #converter em minutos desde a meia noite 
            hor , minuto = float( hor[:2] ) , float( hor[2:] )
            temp = 60*hor + minuto

            #extrair latitude e longitude
            
            try: 
                lat , lon = report[17].split("'")[3] , report[22].split("'")[3]
                coord.append( [ float(lat) , float(lon) ] )
                horas.append( temp )
            except IndexError: pass
            

    horas , coord = numpy.array(horas ) ,numpy.array( coord )
    coord = sklearn.preprocessing.normalize(coord)
    horas = horas.reshape( horas.size , 1 )
    Data = numpy.hstack( ( horas , coord ) )
    Data = sklearn.utils.shuffle( Data )

    return Data

def loadMallData():
    Male , Female = [] , []
    with open( "E:\Datasets\Mall_Customers.csv" , 'r') as Customers:
        Customers.readline()

        for x in range( 200 ):
            line = Customers.readline().split( sep = "," )
            data = [ float(x) for x in line[2:] ]
            gender = line[1]

            if gender == "Male": Male.append( data )
            else: Female.append( data )

    return numpy.array( Male ) , numpy.array( Female )

#EXPERIMENTS WITH KMEANS ------------------------------------------------------------------------------------------------------------------------------------------
def ex1():
    
    Male , Female = loadMallData()

    print("\nmale Clustering" )
    maleCentroids = kCluster( Male , 8 , 120, step = .1 )
    print()

    print("female Clustering" )
    femaleCentroids = kCluster( Female , 8 , 120, step = .1 )

    MaleCluster , FemaleCluster = closestCentroids( Male, maleCentroids ) , closestCentroids( Female, femaleCentroids )
    maleStats , femaleStats = clusterInfo( Male, MaleCluster ) , clusterInfo( Female, FemaleCluster )

    print("\nMaleStats")
    for i in range( 8 ):

        print( "cluster" , i + 1)
        print("Age -> mean" , maleStats[i, 0 ,0 ] , "std" , maleStats[i, 0 ,1 ] )
        print("income -> mean" , maleStats[i, 1 ,0 ] , "std" , maleStats[i, 1 ,1 ] )
        print("spending -> mean" , maleStats[i, 2 ,0 ] , "std" , maleStats[i, 2 ,1 ] )
        print()


    print("FemaleStats")
    for i in range( 8 ):

        print( "cluster" , i + 1)
        print("Age -> mean" , femaleStats[i, 0 ,0 ] , "std" , femaleStats[i, 0 ,1 ] )
        print("income -> mean" , femaleStats[i, 1 ,0 ] , "std" , femaleStats[i, 1 ,1 ] )
        print("spending -> mean" , femaleStats[i, 2 ,0 ] , "std" , femaleStats[i, 2 ,1 ] )
        print()

def ex2():
    
    Data = loadLA_Car_Accidents()

    #spliting data into train and test set---------------------------------------------------------------------------------
    edge = int( .7*Data.shape[0] )
    trainTam , testTam = edge , Data.shape[0] - edge
    trainData , testData = Data[ : trainTam ] , Data[ trainTam : ]

    # Trainig and testing ------------------------------------------------------------------------------------------------
    n = BestVal( Data[:200] , 20 )
    print("Treinando com o numero de clusters escolhido")
    print()
    centroids = kCluster( trainData , n , 200 , 10 , .1 )
    clusters = closestCentroids( testData , centroids )

    cost = 0
    for i in range( 10 ):
        pos = numpy.where( clusters == i )
        members = testData[ pos ]
        D = ( members - centroids[i] )**2
        cost += D.sum()
    
    cost /= testTam

    print("\nCost at testSet: ", cost )

def ex3():
    '''
    given extra weight to the temp column of the LA Accidents, and see what happens 
    '''

    Data = loadLA_Car_Accidents()

    edge = int( .7*Data.shape[0] )
    trainTam , testTam = edge , Data.shape[0] - edge
    trainData , testData = Data[ : trainTam ] , Data[ trainTam : ]

    print()
    horas = trainData[ : , 0 ]
    print( " weightning the time of the accident " )
    for n in range( 1 , 11 ):
        
        print( "weight = ", n, )
        trainData[ : , 0 ] = n*horas

        centroids = kCluster( trainData , 10 , 100 , 0 )
        clusters = closestCentroids( testData , centroids )
        print( kCost(testData , clusters , centroids ) ) 
    
    trainData[ : , 0 ] = horas
    print()

    coord = trainData[ : , 1:3 ]
    print( " weightning the coord of the accident " )
    for n in range( 1 , 11 ):
        
        print( "Taining with weight = ", n, " for the coordinates of the accident")
        trainData[ : , 1:3] = n*coord

        centroids = kCluster( trainData , 10 , 100 , 0 )
        clusters = closestCentroids( testData , centroids )
        print( kCost(testData , clusters , centroids ) ) 
    
    

if __name__ == "__main__":

    ex1()
    ex2()
    ex3()

