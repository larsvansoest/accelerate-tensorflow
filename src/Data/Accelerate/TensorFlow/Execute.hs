{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE InstanceSigs #-}
module Data.Accelerate.TensorFlow.Execute where
import Data.Array.Accelerate.AST.Schedule.Sequential
import Data.Accelerate.TensorFlow.Kernel
import Data.Array.Accelerate hiding (Vector, Exp)
import Data.Array.Accelerate.AST.LeftHandSide
import Data.Text.Prettyprint.Doc (viaShow)
import Data.Array.Accelerate.Pretty.Type (prettyScalarType)
import Data.Array.Accelerate.AST.Kernel
import Data.Array.Accelerate.Interpreter

import qualified TensorFlow.Ops                                     as TF
import qualified TensorFlow.Core                                    as TF
import qualified TensorFlow.Tensor                                  as TF
import qualified TensorFlow.Types                                   as TF
import qualified TensorFlow.Session                                 as TF
import Data.Vector (Vector)
import Data.Int
import qualified TensorFlow.GenOps.Core                             as TF hiding (shape, placeholder)
import Data.IORef
import Data.Array.Accelerate.Array.Buffer
import Data.Array.Accelerate.AST.Environment
import Data.Array.Accelerate.Representation.Type

-- instance Execute SequentialSchedule TensorKernel where
--   executeAfunSchedule = undefined

-- Ik heb geprobeerd naar native execute te kijken, en ik denk dat dit de approach is,
-- het voelt alleen wel of er heel veel misschien al door de interpreter gedaan kan worden
-- kloppen de types, mis ik nog iets?

-- buffers staan in 'env', die sla ik op als tensors
-- scalars staan in 'env' (array groottes, if-else condities), die sla ik op als scalars
--type mijnEnv :: Env _ env
type TensorEnv = Env TensorValue

data TensorValue a where
  TVScalar :: a -> TensorValue a
  TVTensor :: IORef (Either (TF.Tensor TF.Build a) (Vector a)) -> TensorValue (Buffer a)
  -- misschien undefined omdat bij alloc je een placeholder nodig hebt (TF.placeholder?)

type TensorValues = TupR TensorValue

executeSequentialSchedule :: SequentialSchedule TensorKernel env t -> IO ()
executeSequentialSchedule = \case
  SequentialLam lhs ss -> undefined
  SequentialBody ss -> executeSeqSchedule ss

-- evalArrayInstr :: NativeEnv env -> EvalArrayInstr (ArrayInstr env)
-- evalArrayInstr env = EvalArrayInstr $ \instr arg -> case instr of
--   Index buffer -> indexBuffer (groundRelt $ varType buffer) (prj (varIdx buffer) env) arg
--   Parameter (Var tp ix) -> prjGroundVar (Var (GroundRscalar tp) ix) env

executeSeqSchedule :: SeqSchedule TensorKernel env t -> IO (TensorValues t)
executeSeqSchedule = \case  
  Exec km okf pa -> executeKernel km okf pa
  Return tr -> undefined
  Compute exp -> return $ evalExp exp _
  Alet lhs tr ss ss' -> do executeSeqSchedule ss
                           executeSeqSchedule ss'
  Alloc sr st tr -> do putStrLn $ "alloc"
  Use st n bu -> do putStrLn $ "use"
  Unit var -> undefined
  Acond var ss ss' -> undefined
  Awhile tr ssf ssf' tr' -> undefined

executeKernel :: NoKernelMetadata f -> KernelFun TensorKernel args -> SArgs env args -> IO ()
executeKernel m k a = case k of
  KernelFunLam kar okf -> case okf of
    KernelFunLam kar' okf' ->  do putStrLn "executeKernel0"
                                  executeKernelFunLam  kar' okf'
    KernelFunBody tk -> do putStrLn "executeKernel1"
                           executeKernel' tk
  KernelFunBody tk -> do putStrLn "executeKernel2"
                         executeKernel' tk

executeKernelFunLam :: KernelArgR t2 r1 -> OpenKernelFun TensorKernel (r, r1) f2 -> IO ()
executeKernelFunLam kar okf = case okf of
  KernelFunLam kar' okf' -> do putStrLn "KernelFunLam0"
                               executeKernelFunLam kar' okf'
  KernelFunBody tk -> executeKernel' tk

executeKernel' :: TensorKernel a -> IO ()
executeKernel' = \case
  TensorConstant sr st tr s var -> do putStrLn $ "TensorConstant"
  TensorPrimFun sr pf tr tr' tr2 -> do putStrLn $ "TensorPrimFun"
  TensorId sr st tr var var' -> do putStrLn $ "TensorId"
