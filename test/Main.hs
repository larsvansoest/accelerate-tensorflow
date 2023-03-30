{-# LANGUAGE TypeApplications #-}
import Data.Array.Accelerate.Trafo
import Data.Array.Accelerate hiding (Vector)
import Prelude hiding (map, zipWith, (+))
import Data.Accelerate.TensorFlow.Kernel

import qualified TensorFlow.Ops                                     as TF
import qualified TensorFlow.Core                                    as TF
import qualified TensorFlow.Tensor                                  as TF
import qualified TensorFlow.Types                                   as TF
import Data.Vector (Vector)
import Data.Int
import qualified TensorFlow.GenOps.Core                             as TF hiding (shape)
import Data.Array.Accelerate.AST.Schedule.Sequential
import Data.Array.Accelerate.Pretty.Schedule.Sequential
import Data.Array.Accelerate.Pretty
import Data.Array.Accelerate.Pretty.Schedule
import Data.Accelerate.TensorFlow.Execute
import Data.Array.Accelerate.AST.Environment
import Data.Array.Accelerate.Representation.Type (TupR(TupRunit, TupRsingle))
import Control.Monad.IO.Class
import Control.Concurrent
import Data.Array.Accelerate.Array.Buffer
import GHC.MVar

type Tensor = TF.Tensor TF.Build

vectorX :: IO (Vector Int64)
vectorX = do TF.runSession $
              do TF.run $ TF.constant (TF.Shape [2,2,2]) [0, 10, 3, 3, 0, 10, 3, 3]

-- Backend type class

main :: IO ()
main = do putStrLn try
          let sched = convertAfun @SequentialSchedule @TensorKernel $ map @DIM1 @Int64 (\x -> (x + 1) * 2 - (abs (-6))) (use (fromList (Z :. 10) [0..]))
          putStrLn $ renderForTerminal $ prettySchedule sched
          let inputTensorValues = undefined :: TensorElements (MVar (((), Int), Buffer Int64))
          executeSequentialSchedule Empty sched inputTensorValues

-- try = test @UniformScheduleFun @TensorKernel $ x
-- try = let zeros :: Acc (Matrix Int) 
--           zeros = use $ fromList (Z :. 2 :. 3) [0..]
--           ones :: Acc (Matrix Int) 
--           ones = use $ fromList (Z :. 2 :. 3) [1..]
--           in test @UniformScheduleFun @TensorKernel $ permute (+) zeros Just_ ones


  -- let zeros :: Acc (Vector Int) 
  --         zeros = fill (constant (Z:.10)) 0
  --         ones :: Acc (Vector Int) 
  --         ones = fill (constant (Z:.10)) 0
  --         in test @UniformScheduleFun @TensorFlowKernel $ permute (+) zeros Just_ ones

try :: String
--try = test @SequentialSchedule @TensorKernel $ map @DIM1 @Int (\x -> (x + 1) * 2) (use (fromList (Z :. 10) [0..]))
try = test @SequentialSchedule @TensorKernel $ map @DIM1 @Int64  (\x -> (x + 1) * 2) (use (fromList (Z :. 10) [0..]))

-- try =  test @UniformScheduleFun @TensorKernel $ zipWith @DIM1 @Int (+) (use (fromList (Z :. 10) [0..])) (use (fromList (Z :. 10) [0..]))

 -- test @UniformScheduleFun @InterpretKernel $ zipWith @DIM1 @Int (+)