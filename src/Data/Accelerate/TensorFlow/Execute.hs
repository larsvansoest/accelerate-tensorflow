{-# LANGUAGE BangPatterns      #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs             #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies      #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Use record patterns" #-}
{-# LANGUAGE LambdaCase #-}
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
import Data.Array.Accelerate.AST.Environment hiding (push)
import Data.Array.Accelerate.Representation.Type
import Data.Array.Accelerate.Type
import Data.Array.Accelerate.Representation.Ground
import Data.Array.Accelerate.Analysis.Match

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
  TensorScalar :: a -> TensorValue a
  TensorBuild :: IORef (Either (TF.Tensor TF.Build a) (Vector a)) -> TensorValue (Buffer a)
  TensorValue :: IORef (Vector a) -> TensorValue (Buffer a)
  -- misschien undefined omdat bij alloc je een placeholder nodig hebt (TF.placeholder?)

type TensorValues = TupR TensorValue

executeSequentialSchedule :: SequentialSchedule TensorKernel env t -> IO ()
executeSequentialSchedule (SequentialLam lhs sched) = undefined
executeSequentialSchedule (SequentialBody sched)    = undefined -- executeSeqSchedule sched

executeSeqSchedule :: TensorEnv env -> SeqSchedule TensorKernel env t -> IO (TensorValues t)
executeSeqSchedule env (Exec m fun args) = executeKernel env m fun args
executeSeqSchedule env (Return tr) = undefined
executeSeqSchedule env (Compute expr) = return $ TupRsingle (TensorScalar (evalExp expr $ evalArrayInstr env))
executeSeqSchedule env (Alet lhs _ sched sched') = do
  rhs <- executeSeqSchedule env sched 
  let env' = push env (lhs, rhs)
  executeSeqSchedule env' sched'
executeSeqSchedule _ (Alloc sr st tr) = undefined
executeSeqSchedule _ (Use st n bu) = undefined
executeSeqSchedule _ (Unit var) = undefined
executeSeqSchedule _ (Acond var ss ss') = undefined
executeSeqSchedule _ (Awhile tr ssf ssf' tr') = undefined

push :: TensorEnv env -> (LeftHandSide s t env env', TensorValues t) -> TensorEnv env'
push env (LeftHandSideWildcard _, _)            = env
push env (LeftHandSideSingle _  , TupRsingle a) = env `Push` a
push env (LeftHandSidePair l1 l2, TupRpair a b) = push env (l1, a) `push` (l2, b)
push _ _                                        = error "Tuple mismatch"

executeKernel :: TensorEnv env -> NoKernelMetadata f -> KernelFun TensorKernel args -> SArgs env args -> IO (TensorValues t)
executeKernel _ _ (KernelFunLam _ _) _ = undefined
executeKernel _ _ (KernelFunBody _) _  = undefined

executeKernelFunLam :: KernelArgR t2 r1 -> OpenKernelFun TensorKernel (r, r1) f2 -> IO ()
executeKernelFunLam kar okf = case okf of
  KernelFunLam kar' okf' -> do putStrLn "KernelFunLam0"
                               executeKernelFunLam kar' okf'
  KernelFunBody tk -> executeKernel' tk

executeKernel' :: TensorKernel t -> IO (TensorValues t)
executeKernel' = \case
  TensorConstant sr st tr s var -> do putStrLn $ "TensorConstant"
  TensorPrimFun sr pf tr tr' tr2 -> do putStrLn $ "TensorPrimFun"
  TensorId sr st tr var var' -> do putStrLn $ "TensorId"

evalArrayInstr :: TensorEnv env -> EvalArrayInstr (ArrayInstr env)
evalArrayInstr env = EvalArrayInstr $ \instr arg -> case instr of
  Index buffer -> indexBuffer (groundRelt $ varType buffer) (prj' (varIdx buffer) env) arg
  Parameter (Var _ idx) -> prjGroundVar (Var (GroundRscalar tp) ix) env

