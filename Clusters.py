import numpy


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
        if ( it + 1)%clock == 0:
            custo = numpy.sum( numpy.array( [ Data[i] - centroids[ clusters[i] ] for i in range(x) ] )**2 )/x
            print( "custo",( it + 1),"=", custo , sep = " ")

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


if __name__ == "__main__":

    Male , Female = [] , []
    with open( "E:\Datasets\Mall_Customers.csv" , 'r') as Customers:
        Customers.readline()

        for x in range( 200 ):
            line = Customers.readline().split( sep = "," )
            data = [ float(x) for x in line[2:] ]
            gender = line[1]

            if gender == "Male": Male.append( data )
            else: Female.append( data )
    
    Male , Female = numpy.array( Male ) , numpy.array( Female )

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

            


