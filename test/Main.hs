{-# LANGUAGE TypeApplications #-}
import Data.Array.Accelerate.Trafo
import Data.Array.Accelerate
import Data.Array.Accelerate.Interpreter
import Prelude hiding (map, zipWith)
import Data.Accelerate.TensorFlow.Operation (TensorFlowKernel)

main :: IO ()
main = putStrLn try

try :: String
try = test @UniformScheduleFun @TensorFlowKernel $ map @DIM1 @Int (+ 1 * 2) (use (fromList (Z :. 10) [0..]))

-- test @UniformScheduleFun @InterpretKernel $ zipWith @DIM1 @Int (+)