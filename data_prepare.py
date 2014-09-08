# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------
# Name:        data_prepare.py
# Author:      Yandi LI
# Created:     2014/5/2
# Objective:   Create the dataset for FM-BPR() model, save in pickle format 
# Use:         python *.py [options] <outfile>
# Require :    table:context_cube, table:checkin_venue
#-----------------------------------------------------------------------------
import psycopg2, sys
from numpy import array
from scipy.sparse import csr_matrix
from numpy import unique
from optparse import OptionParser


def query_table(con, query):
  """
  send a query to the database, return a list of incidences 
  """
  cur = con.cursor()
  cur.execute(query)
  result = cur.fetchall()
  con.commit()
  return result


def create_data(options, context_features = ['hour','conds','dow']): 
  try:
    con = psycopg2.connect(database=options.database,
                           host=options.host, 
                           user=options.user,
                           password=options.password)    

    query = 'SELECT (' +\
              'row_number() OVER ())-1 AS cid,' +\
              ', '.join(context_features) +\
            ' FROM '+ options.tb_name_c +\
            ' GROUP BY '+ ', '.join(context_features)+\
            ' ORDER BY cid'
    A = query_table(con, query) # cid-feature_vector matrix

    query = 'WITH cid_attr as (\
            SELECT (row_number() OVER ())-1 AS cid,'+\
            ', '.join(context_features) +\
            ' FROM '+ options.tb_name_c +\
            ' GROUP BY ' + ', '.join(context_features) +\
            ')' +\
            """SELECT cid, vid, COUNT(tweetid)::int AS count 
            FROM """ + options.tb_name_c + """ a
              NATURAL JOIN cid_attr c
            GROUP BY cid, vid
            ORDER BY cid"""   
    Y = query_table(con,query)  # cid-vid-count matrix
     
    query = 'SELECT DISTINCT vid, tid'+\
            ' FROM ' + options.tb_name_v +\
            ' ORDER BY vid'
    T = query_table(con, query) # vid-tid matrix

  except psycopg2.DatabaseError, e:
    print 'Error %s' % e    
    sys.exit(1)
  else:
    print "Loading database: Success!"
  finally:
    if con:
      con.close()

  Y = array(Y)
  cid_attributes = array(A)[:,1:] # omit index column
  vid_attributes = array(T,dtype='int')[:,1:] # omit index column

  data = {'triple':csr_matrix((Y[:,2],(Y[:,0],Y[:,1]))),\
          'c_attr':cid_attributes,\
          'v_attr':vid_attributes}
  return data


if __name__ == '__main__':
  import pickle

  ###############################
  ## Argument Parsing
  ###############################
  optparser = OptionParser(usage="""%prog [options] <output_pickle_file>
  Query context_cube table to build design matrices in pickle format
  """)
  defaults = {'host':'localhost',\
              'user':'postgres',\
              'password':'',
              'tb_name_c':'context_cube',  
              'tb_name_v':'checkin_venue',
              'features':'hour conds dow'}
  optparser.set_defaults(**defaults)
  optparser.add_option('-d','--database',dest='database',
                      type='string', help='Database name.')
  optparser.add_option('-H','--host',dest='host',
                      type='string', help='Host address. Default is %s.' % repr(defaults['host']))
  optparser.add_option('-u','--user',dest='user',
                      type='string', help='User name. Default is %s.'% repr(defaults['user']))
  optparser.add_option('-p','--password',dest='password',
                      type='string', help='Password. Default is empty')
  optparser.add_option('-t','--table',dest='tb_name_c',
                      type='string', help='Table name of the context data cube. Default is %s.'%repr(defaults['tb_name_c']))
  optparser.add_option('-v','--vtable',dest='tb_name_v',
                      type='string', help='Table name of the venue checkin table. Default is %s.'%repr(defaults['tb_name_v']))
  optparser.add_option('-f','--features',dest='features',
                      type='string', help='Select columns to include as context features. Default is %s.'%repr(defaults['features']))

  options, (outfile,) = optparser.parse_args()
  if not options.database:
    optparser.error('Database name not given.')

  ###############################
  ## Output to file
  ###############################
  context_features = options.features.split() # select columns to include as context features
  data = create_data(options, context_features)
  pickle.dump(data, open(outfile,'w'))
  print 'Well Done.'

