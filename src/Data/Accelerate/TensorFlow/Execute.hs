{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE InstanceSigs #-}
module Data.Accelerate.TensorFlow.Execute where
import Data.Array.Accelerate.AST.Schedule.Sequential
import Data.Accelerate.TensorFlow.Kernel
import Data.Array.Accelerate hiding (Exp)
import Data.Array.Accelerate.AST.LeftHandSide
import Data.Text.Prettyprint.Doc (viaShow)
import Data.Array.Accelerate.Pretty.Type (prettyScalarType)
import Data.Array.Accelerate.AST.Kernel

-- instance Execute SequentialSchedule TensorKernel where
--   executeAfunSchedule = undefined

executeSequentialSchedule :: SequentialSchedule TensorKernel env t -> IO ()
executeSequentialSchedule = \case
  SequentialLam lhs ss -> undefined
  SequentialBody ss -> executeSeqSchedule ss

executeSeqSchedule :: SeqSchedule TensorKernel env t -> IO ()
executeSeqSchedule = \case  
  Exec km okf pa -> executeKernel km okf pa
  Return tr -> undefined
  Compute exp -> computeExp exp
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

computeExp :: Exp env t -> IO ()
computeExp = \case  
  Let lhs poe poe' -> undefined
  Evar var -> undefined
  Foreign tr asm pof poe -> undefined
  Pair poe poe' -> undefined
  Nil -> undefined
  VecPack vr poe -> undefined
  VecUnpack vr poe -> undefined
  IndexSlice si poe poe' -> undefined
  IndexFull si poe poe' -> undefined
  ToIndex sr poe poe' -> undefined
  FromIndex sr poe poe' -> undefined
  Case poe x0 m_poe -> undefined
  Cond poe poe' poe2 -> undefined
  While pof pof' poe -> undefined
  Const st t -> do putStrLn $ show $ prettyScalarType st
  PrimConst pc -> undefined
  PrimApp pf poe -> undefined
  ArrayInstr ai poe -> undefined
  ShapeSize sr poe -> undefined
  Undef st -> undefined
  Coerce st st' poe -> undefined
