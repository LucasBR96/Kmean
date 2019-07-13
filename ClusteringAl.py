from scipy import stats
import numpy
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.utils
import math

class OGMM:

    def __init__( self, Data , n ):
        '''

        One Dimension Gaussian Mixture Model.

        Arguments:
            Data -> flat numerical numpy Array
            G -> Number of Gaussians

        '''

        self.Data = Data
        self.x = Data.size

        self.n = n
        self.weights = numpy.ones( n )/n #Gaussian weights

        G = numpy.zeros( ( n , 2 ) )
        G[ : , 0 ] = numpy.linspace( numpy.min( Data ) , numpy.max( Data ) , n )
        G[ : , 1 ] = numpy.std( Data )/n
        self.G = G #Gaussian Matrix

    def fit( self, iters = 30, clock = 0 ):
        
        F = numpy.zeros( ( self.x , self.n ) ) #distance matrix
        
        for it in range( iters ):

            #Estep-------------------------------------------------------------------------------------------------------------------------------------------------------------
            den = self.likelihood( self.Data )
            for k in range( self.n ):
                mu , sigma = self.G[ k ]
                w = self.weights[ k ]

                num = w*stats.norm.pdf( self.Data, loc = mu , scale = sigma )
                F[ : , k ] = num/den
            
            #Mstep--------------------------------------------------------------------------------------------------------------------------------------------------------------
            for k in range( self.n ):

                col = F[ : , k]

                #updating means
                den = col.sum()
                self.G[ k , 0 ] = ( ( self.Data*col )/den ).sum()

                #updating deviations
                sigma = ( self.Data - self.G[ k , 0 ] )**2
                self.G[ k , 1 ] =  math.sqrt( ( col*sigma / den ).sum() )

                #updating weights
                self.weights[ k ] = den/self.x
            
            #Checking cost ------------------------------------------------------------------------------------------------------------------------------------------------------
            if clock != 0:
                if ( it + 1 )%clock == 0:
                    custo = self.Cost()
                    print("Custo médio no iter {}: {:.10}".format( it + 1 , custo ))
        
    def Cost( self , seq = None):

        if seq == None: 
            seq = self.Data

        probs = self.likelihood( seq )
        result = -numpy.log( probs ).sum()
        return result/self.x

    def likelihood( self , val ):

        soma = 0
        if type( val ) == numpy.ndarray:
            soma = numpy.zeros( val.shape )
        
        for k in range( self.n ):
            mu , sigma = self.G[ k ]
            w = self.weights[ k ]

            soma += w*stats.norm.pdf( val, loc = mu , scale = sigma )

        return soma


class MGMM:
    
    def __init__( self , Data , n ):
        '''

        Multi Dimension Gaussian Mixture Model.

        Arguments:
            Data -> flat numerical numpy Array
            G -> Number of Gaussians

        '''

        self.Data = Data
        self.x , self.y = Data.shape

        self.n = n
        self.weights = numpy.ones( n )/n #Gaussian weights

        G = numpy.zeros( ( n , self.y ) )
        for y in range( self.y ):
            G[ : , y ] = numpy.linspace( numpy.min( Data[ : , y] ) , numpy.max( Data[ : , y] ) , n )
        
        self.Means = G #means Matrix

        COV = numpy.cov( Data , rowvar = False )
        COV = COV/self.n #Covariance Matrix
        self.COV = numpy.zeros( ( n , self.y ,self.y ) )
        for k in range( n ):
            self.COV[ k ] = COV
        

    def fit( self, iters = 30, clock = 0 , plotCosts = True):

        F = numpy.zeros( (self.x , self.n) )

        for it in range( iters ):

            #Estep-------------------------------------------------------------------------------------------------------------------------------------------------------------
            for k in range( self.n ):

                #computing density function--------------------------------------------------------------------------------------------------------------------------------
                mu = self.Means[k]
                sigma = self.COV[k]

                detSigma = numpy.linalg.det( sigma )
                inv = sigma/detSigma
                
                for i in range( self.x ):
                    line = self.Data[ i ]

                    #numerator
                    cv = ( line - mu ).reshape( self.y , 1)
                    power = -.5*( ( cv.T@inv )@cv )
                    num = numpy.exp( power ) 

                    #denominator
                    Pprod = ( 2*numpy.pi )**self.y
                    den = math.sqrt( Pprod*detSigma )

                    F[ i , k ] = num/den
            
            F = F*self.weights
            Prob = numpy.sum( F , axis = 1 , keepdims = True )
            F = F/Prob

            #Mstep----------------------------------------------------------------------------------------------------------------------------------------------------------------
            for k in range( self.n ):

                col = F[ : , k ]
                den = col.sum()
                mu = self.Means[k].copy()

                #updating means[k]
                self.Means[ k ] = ( self.Data*col.reshape( self.x , 1 )/den ).sum( axis = 0 )

                #Updating Covariance
                COV = numpy.zeros( ( self.y , self.y ) )
                d = self.Data - mu
                for i in range( self.x ):
                    line = d[i].reshape( 1 , self.y )
                    COV += ( line.T@line )*col[i]
                self.COV[k] = COV/den

                #updating weights
                self.weights[k] = den/self.x
            
            #Checking cost ------------------------------------------------------------------------------------------------------------------------------------------------------
            if clock != 0:
                if ( it + 1 )%clock == 0:
                    custo = -( numpy.log( Prob ).sum() )/self.x
                    print("Custo médio no iter {}: {:.10}".format( it + 1 , custo ))
            
            

    def Cost( self , seq = None):
        
        if seq == None: 
            seq = self.Data

        probs = self.likelihood( seq )
        result = -numpy.log( probs ).sum()

        return result/self.x

    def likelihood( self , val ):

        if type( val ) == numpy.ndarray:
            soma = numpy.zeros( val.shape[0] )
        else:  
            soma = numpy.array( [ 0 ] ) 
            val = numpy.array( [ 0 ] )

        for k in range( self.n ):

            #computing density function
            mu = self.Means[k]
            sigma = self.COV[k]
            w = self.weights[k]
            detSigma = numpy.linalg.det( sigma )
            inv = sigma/detSigma


            for i in range( val.size ):
                x = val[i]
                line = x.reshape( self.y , 1 )

                #numerator
                cv = line - mu
                power = -.5*( ( cv.T@inv )@cv )
                num = numpy.exp( power ) 

                #denominator
                
                Pprod = ( 2*numpy.pi )**2
                den = math.sqrt( Pprod )

                soma[i] += w*num/den

        if val.size == 1:
            return soma[ 0 ]
        else:
            return soma

    

#LOADING DATA SCRIPTS------------------------------------------------------------------------------------------------------------------------------------------------------------
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
    # coord = sklearn.preprocessing.normalize(coord)
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

#EXAMPLES------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def ex2():
    
    Data = numpy.vstack( loadMallData() )
    # Data = Data[ : , ( 0 , 2 ) ]

    M1 = MGMM( Data , 4 )
    M1.fit( iters = 100 , clock = 2)


def ex4():

    Data = loadLA_Car_Accidents()
    Horas = Data[ : , 0 ]
    numpy.random.shuffle( Horas )

    plt.hist( Horas , 96, density = True )
    plt.show()

    M1 = OGMM( Horas , 10 )
    M1.fit( iters = 50 , clock = 10 )

    x = numpy.array( list( range( 24*60 ) ) )
    y = M1.likelihood( x )

    plt.plot( x , y )
    plt.show()

if __name__ == "__main__":
    ex2()