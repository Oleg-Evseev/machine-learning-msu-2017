#!/usr/bin/env python

import functools# return altered function, etc.
import math# log, exp
import numpy as np# arrays, etc.
import sys# print without spaces
from termcolor import colored# colours while printing

# main constants
TEST_MODE = False
#TEST_MODE = True
PROGRAM_NAME = "program 1"
TYPE_COL = 29
DATA_FILE =    "../3fgl_full.dat"

#************** Building Tree ****************
# this function splits the array 'D' by the median of the 'column''s column
def splitBranch( D , column ) :
	# 'D' is the array of data, 'column' is a number of column to sort with

	# checks for column to be normal
	if column < 0 :
		print "Error: column =", column
		column = 0
	if column >= len( D [ 0 ] ) :
		print "Error: column =", column
		column = len( D [ 0 ] ) - 1

	# take the column adn get median
	D_col = D [ : , column ]
	m = np.median( D_col )

	return [
		D [ np.where( D_col < m ) ] ,
		D [ np.where ( D_col >= m ) ] ,
		[ column , m ]
	]

# this function builds the tree itself
def buildTree( D , columns ) :
	# 'D' is the array of data, 'columns' is a list of columns to sort by

	# if no columns to split
	if not columns :
		return D

	# create local copy of columns and take the first, also delete it from local array
	columns_local = columns [ : ]
	col = columns_local . pop( 0 )

	T = splitBranch( D , col )

	# if not splitted, this is the leaf
	if ( len ( T [ 0 ] ) < 1 ) or ( len ( T [ 1 ] ) < 1 ) :
		return D

	return [
		buildTree( T [ 0 ] , columns_local ) ,
		buildTree( T [ 1 ] , columns_local ) ,
		T [ 2 ]
	]

# this function sorts the row using the existing tree
def sortWithTree( T , row ) :
	# 'T' is the tree, 'row' is a single row of data
	if ( len( T ) == 3 ) and ( len( T [ 2 ] ) == 2 ) :
		column = T [ 2 ] [ 0 ]
		median = T [ 2 ] [ 1 ]

		if row[ column ] < median :
			return sortWithTree( T [ 0 ] , row )

		return sortWithTree( T [ 1 ] , row )

	#if we are here then there is no tree structure anymore, we are at the final level
	types = T [ : , TYPE_COL ]
	pulsar = len( T [ np.where( types == 1 ) ] )
	blazar = len( T [ np.where( types == -1 ) ] )
	return 1 if pulsar > blazar else -1

# this function returns the classifier -- the function which takes only the row for the parameter and returns its classification
def getClassifier ( D_TRAIN, col_range ) :
	# build a tree

	DTree = buildTree( D_TRAIN , col_range )

	#the first argument is the name of the function, the second is the first parameter of it. We return the function with a predefined first parameter
	return functools.partial( sortWithTree , DTree )


#************** Classic Boosting *************
# this function returns the right classifier depending on the results of the first two
def boostClassifierChoice ( CL1 , CL2 , CL3 , row ) :
	return CL1( row ) if CL1( row ) == CL2 ( row ) else CL3( row )

# the function of testing the classifier
def testClassifier ( CL , D_TESTS ) :
	# run the classificator for each row in the training set 
	D_TEST_RES = np.array( map( CL , D_TESTS ) )

	return [
		len( D_TESTS [ np.where( D_TESTS [ : , TYPE_COL ] != D_TEST_RES ) ] ),# number of errors
		len( D_TESTS ),# number of tests
		len( D_TESTS [ np.where( ( D_TESTS [ : , TYPE_COL ] != D_TEST_RES ) & ( D_TESTS [ : , TYPE_COL ] == -1 ) ) ] ),#number of blazar errors
		len( D_TESTS [ np.where( D_TESTS [ : , TYPE_COL ] == -1 ) ] ),# number of blazars
		len( D_TESTS [ np.where( ( D_TESTS [ : , TYPE_COL ] != D_TEST_RES ) & ( D_TESTS [ : , TYPE_COL ] ==  1 ) ) ] ),# number of pulsar errors
		len( D_TESTS [ np.where( D_TESTS [ : , TYPE_COL ] ==  1 ) ] )# number of pulsars
	]


# the function of boosting itself
def boostClassifierOnce ( CL1 , D_TRAIN , col_range ) :
	# test the classifier
	D_TEST_RES_1 = np.array( map( CL1 , D_TRAIN ) )

	# get all errors
	D_ERROR = D_TRAIN [ np.where ( D_TRAIN [ : , TYPE_COL ] != D_TEST_RES_1 ) ]
	error_count = len ( D_ERROR )

	if error_count < 1 :
		return [ CL1 , False ]

	# get the same number of correct
	D_CORRECT_TMP = D_TRAIN [ np.where ( D_TRAIN [ : , TYPE_COL ] == D_TEST_RES_1 ) ]
	np.random.shuffle ( D_CORRECT_TMP )
	D_CORRECT = D_CORRECT_TMP [ : error_count ]

	# new training set with all the errors and the same number of correct
	D_TRAIN_NEW = np.concatenate( ( D_ERROR , D_CORRECT ) , axis=0 )

	# build the second classifier
	CL2 = getClassifier( D_TRAIN_NEW , col_range )


	# test it
	D_TEST_RES_2 = np.array( map( CL2 , D_TRAIN ) )

	# get where the two classifiers act differently
	D_DIFFERENCE = D_TRAIN [ np.where( D_TEST_RES_1 != D_TEST_RES_2 ) ]

	if len( D_DIFFERENCE ) < 2 :
		return [ CL1 , False ]

	# build the second classifier
	CL3 = getClassifier( D_DIFFERENCE , col_range )

	# test it on its own training set
	test_res = testClassifier ( CL3 , D_DIFFERENCE )

	# if it helps with less than a half it's useless
	if test_res [ 1 ] >= 0.5 * test_res [ 0 ] :
		return [ CL1 , False ]

	return [ functools.partial( boostClassifierChoice , CL1 , CL2 , CL3 ) , True ]


# the cycle of boosting
def boostClassifier ( CL1 , D_TRAIN , col_ranges ) :
	Proceed = True
	CL = CL1
	counter = -1;# the first iteration of boosting is taken anyway, if it fails the counter becomes 0, which is correct
	col_ranges_local = col_ranges [ : ]

	while Proceed :
		# define the column range for boosting
		if len ( col_ranges_local ) == 1 :
			col_range = col_range [ 0 ]
		else :
			col_range = col_ranges_local . pop ( 0 )

		Boosted = boostClassifierOnce ( CL , D_TRAIN , [ col_range ] )
		Proceed = Boosted [ 1 ]
		CL = Boosted [ 0 ]
		counter += 1

	return [ CL , counter ]


#************** Adaptive Boosting ************

# calculate the weighted error for the classificator 'CL'
def adaEpsilon ( w , D_TEST , CL ) :
	D_TEST_RES = np.array( map( CL , D_TEST ) )

	# the condition given returns the array of Trues and Falses which taken as integers is an array of 1 and 0 correspondingly
	# than we reduce the ziped array of pairs -- multiply each weight by the test result
	# finaly we reduce by summing the rest dimention
	return np.add.reduce( np.multiply.reduce( zip( w , ( D_TEST [ : , TYPE_COL ] != D_TEST_RES ).astype( int ) ) , axis = 1  ) )


def adaLinCombine ( CLS , alphas , row ) :
	#create the list of identical rows -- needed for map() function
	rows = np.tile( row , ( len (CLS) , 1 ) )

	# apply each classifier from the list 'CLS' to the corresponding row in 'rows'
	CLS_RES = map( lambda f, row: f(row) , CLS , rows )

	return -1 if ( np.add.reduce( np.multiply.reduce( zip( CLS_RES , alphas ) , axis = 1  ) ) < 0 ) else 1

def adaBoostClassifiers ( CLS , D_TRAIN ) :
	# 'CLS' is a set of classifiers , 'D_TRAIN' is a training set

	# number of training objects
	m = len ( D_TRAIN )

	# number of classifiers
	T = len ( CLS )

	# create the array of weights and fill it with 1/m
	w = np.empty ( m )
	w.fill ( 1.0 / m )

	CL_MIN = []
	alpha  = []

	# run the infinite cycle -- it will end with return when the improvement will stop
	for t in range( T ):
		# apply to each classifier in 'CLS' the fuction that calculates epsilon with predefinded 'w' and 'D_TRAIN'
		epsilons = np.array( map ( functools.partial( adaEpsilon , w , D_TRAIN ) , CLS ) )

		# take the classifier with minimal epsilon
		epsilon_min_index = np.argmin( epsilons )
		epsilon_min = epsilons [ epsilon_min_index ]
		CL_MIN.append( CLS [ epsilon_min_index ] )

		# if the best classifier is bad
		if epsilon_min >= 0.5 :
			# return the improved classifier if it is set, CL_MIN otherwize
			alpha.append( 0 );
			continue

		# if the best classifier is perfect
		if epsilon_min == 0.0 :
			return CL_MIN [ -1 ]

		# run the test of this classificator, get the data, we'll need it
		D_TEST_RES = np.array( map( CL_MIN [ -1 ] , D_TRAIN ) )

		# calculate the exponential parameter
		alpha.append( 0.5 * math.log ( ( 1 - epsilon_min ) / epsilon_min ) )

		# update weights:
		# x [ 0 ] is the previous weight
		# it should be multiplied by exp( alpha ) if error or by exp( - alpha ) if correct
		# the above algorythm is given when mutiplying by exp ( - alpha * <real_type> * <classified_type> )
		#
		# please note that row [ 0 ] is current 'w' , row [ 1 ] is current row of 'D_TRAIN' , row [ 2 ] is current element of 'D_TEST_RES'
		w = map ( lambda row : row [ 0 ] * math.exp( - alpha [ -1 ] * row [ 1 ] [ TYPE_COL ] * row [ 2 ] ) , zip ( w , D_TRAIN , D_TEST_RES ) )

		# normalize 'w'
		norm = np.add.reduce( w )
		w = np.array ( map ( lambda x: x / norm , w ) )

	# return the resulting lineal combination of classifiers:
	# adaLinCombine takes 3 arguments: the list of Classifiers, the list of alphas and a row of data
	# we reduce it to only the last one making the returning result a true classifier -- function of one parameter
	return functools.partial( adaLinCombine , CLS , alpha )


#************** Formated Output **************
# this function prints the number of errors and its percentage from total number of tests
def printTestResults ( msg , errors, total ) :
	print msg , errors , "errors" if ( errors != 1 ) else "error" , "out of" , total , "tests (" , 100.0 * errors / total , "% )"

# this function tests the classifier CL on the test set D_TESTS given
def printTestClassifier ( msg , CL , D_TESTS ) :
	print "\nTesting the classificator \"" + msg + "\""
	# run the classificator for each row in the training set 
	sys.stdout.write( "Running on the training set..." )
	test_results = testClassifier( CL , D_TESTS )
	sys.stdout.write( "   Done\n" )

	printTestResults ( "Test results:" , test_results [ 0 ] , test_results [ 1 ] )
	printTestResults ( "Blazars:"      , test_results [ 2 ] , test_results [ 3 ] )
	printTestResults ( "Pulsars:"      , test_results [ 4 ] , test_results [ 5 ] )

	print "Test completed\n"


#************** Active Code ******************

print colored( "Starting " + PROGRAM_NAME , 'green' )

# load data file
sys.stdout.write( "Loading file..." )
D3 = np.loadtxt( DATA_FILE , dtype = 'string' )
sys.stdout.write( "   Done\n" )

# get the 30th column
tp = D3 [ : , TYPE_COL ]

# identify pulsars as 1, other known as -1
D3 [ np.where( ( tp == 'bll' ) | ( tp == 'BLL' ) | ( tp == 'bcu' ) | ( tp == 'BCU' ) | ( tp == 'fsrq' ) | ( tp == 'FSRQ' ) ) , TYPE_COL ] = -1
D3 [ np.where( ( tp == 'psr' ) | ( tp == 'PSR' ) ) , TYPE_COL ] = 1

#form a train set from half of known
D_TRAIN_FULL = D3 [ np.where( ( D3 [ : , TYPE_COL ] == '-1' ) | ( D3 [ : , TYPE_COL ] == '1' ) ) ]

if TEST_MODE :
	D_TRAIN = D_TRAIN_FULL [ 0 :: 2 ]# not a random set but each even -- useful for testing
	D_TESTS = D_TRAIN_FULL [ 1 :: 2 ]# not a random set but each odd -- useful for testing
else :
	rnd_set = np.random.choice( [ 0 , 1 ] , len( D_TRAIN_FULL ) )
	D_TRAIN = D_TRAIN_FULL [ np.where( rnd_set == 0 ) ]
	D_TESTS = D_TRAIN_FULL [ np.where( rnd_set == 1 ) ]

# delete the first row information (replace by 1)
D_TRAIN [ : , 0 ] = 1
D_TESTS [ : , 0 ] = 1

# convert to float
D_TRAIN = D_TRAIN.astype('float')
D_TESTS = D_TESTS.astype('float')

print "Known elements:" ,  len( D_TRAIN ) + len( D_TESTS ) , "= (" , len(D_TRAIN), "for training ) + (" , len(D_TESTS) , "for tests )"

print "Train elements (" , len ( D_TRAIN ) , "total ):" ,  len( D_TRAIN [ np.where( D_TRAIN[ : , TYPE_COL ] == -1 ) ] ) , "blazars and" , len( D_TRAIN [ np.where( D_TRAIN[ : , TYPE_COL ] == 1 ) ] ) , "pulsars"
print "Test elements  (" , len ( D_TESTS ) , "total ):" ,  len( D_TESTS [ np.where( D_TESTS[ : , TYPE_COL ] == -1 ) ] ) , "blazars and" , len( D_TESTS [ np.where( D_TESTS[ : , TYPE_COL ] == 1 ) ] ) , "pulsars"

print "\nTesting the splitBranch function:"

sys.stdout.write( "Building two branches by the " + str( TYPE_COL ) + "th column..." )
DT1 = splitBranch( D_TRAIN , 28 )
sys.stdout.write( "   Done\n" )

print "Results of splitting:"
print "-------------"
print "|" , len( np.where( DT1 [ 0 ] [ : , TYPE_COL ] == -1 ) [ 0 ] ) , "| " , len( np.where( DT1 [ 0 ] [ : , TYPE_COL ] == 1 ) [ 0 ] ) , "|"
print "-------------"
print "|" , len( np.where( DT1 [ 1 ] [ : , TYPE_COL ] == -1 ) [ 0 ] ) , "| " , len( np.where( DT1 [ 1 ] [ : , TYPE_COL ] == 1 ) [ 0 ] ) , "|"
print "-------------"


print "\nTrying classical boosting:"
col_ranges = [
	range (  3 , 13 ),
	range ( 18 , 18 ),
	range ( 13 , 23 ),
	range ( 18 , 28 )
]

CL = getClassifier( D_TRAIN , col_ranges [ 0 ] )
printTestClassifier( "CL" , CL , D_TESTS )

Boosted = boostClassifier( CL , D_TRAIN , col_ranges )

CL_B = Boosted [ 0 ]

print "After" , Boosted [ 1 ] , "boosts:"

printTestClassifier( "CL_B" , CL_B , D_TESTS )


print "\nTrying adaptive boosting (AdaBoost):"

CLS = [
	getClassifier( D_TRAIN [ : ] , col_ranges [ 0 ] ),
	getClassifier( D_TRAIN [ : ], col_ranges [ 1 ] ),
	getClassifier( D_TRAIN [ : ], col_ranges [ 2 ] ),
	getClassifier( D_TRAIN [ : ], col_ranges [ 3 ] )
]

counter = 1

CL_AB = adaBoostClassifiers( CLS , D_TESTS )

printTestClassifier( "CL_AB" , CL_AB , D_TESTS )

print colored ( "Done" , "green" )
