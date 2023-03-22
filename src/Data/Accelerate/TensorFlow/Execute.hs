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
import Control.Monad.IO.Class
import Unsafe.Coerce
import Data.Accelerate.TensorFlow.Tensor hiding (toBuffer)
import Data.Array.Accelerate.Representation.Shape
import Data.Array.Accelerate.Trafo.Operation.Substitution

-- instance Execute SequentialSchedule TensorKernel where
--   executeAfunSchedule = undefined

executeSequentialSchedule :: TensorEnv env -> SequentialSchedule TensorKernel env t -> IO ()
executeSequentialSchedule env (SequentialLam lhs sched) = executeSequentialSchedule (_ env) sched -- ?
executeSequentialSchedule env (SequentialBody sched)    = do
  _ <- executeSeqSchedule env sched
  return () -- hoe moet ik van TensorValues t naar ()?

executeSeqSchedule :: TensorEnv env -> SeqSchedule TensorKernel env t -> IO (TensorValues t)
executeSeqSchedule env (Exec m fun args) = executeKernel env m fun args

executeSeqSchedule env (Return tr) = return $ prjAll env tr
  where prjAll :: TensorEnv env -> GroundVars env t -> TensorValues t
        prjAll _ TupRunit = TupRunit
        prjAll env (TupRsingle (Var _ idx)) = TupRsingle (prj' idx env)
        prjAll env (TupRpair v v') = TupRpair (prjAll env v) (prjAll env v')

executeSeqSchedule env (Compute expr) =
  return (TupRsingle (TensorScalar (evalExp expr (evalArrayInstr env))))

executeSeqSchedule env (Alet lhs _ sched sched') = do
  rhs <- executeSeqSchedule env sched
  let env' = push env (lhs, rhs)
  executeSeqSchedule env' sched'

executeSeqSchedule _ (Alloc shR st vars) = return $ TupRsingle $ TensorValue $ liftIO (TF.placeholder _) -- ?

executeSeqSchedule _ (Use st n buffer) = return $ TupRsingle (TensorBuild' (fromBuffer dim1 st ((), n) buffer))

executeSeqSchedule _ (Unit var) = undefined

executeSeqSchedule _ (Acond var ss ss') = undefined

executeSeqSchedule _ (Awhile tr ssf ssf' tr') = undefined

executeKernel :: TensorEnv env -> NoKernelMetadata f -> KernelFun TensorKernel args -> SArgs env args -> IO (TensorValues t)
executeKernel env m (KernelFunLam z kernel) args = undefined --executeKernel (_ env) m kernel (_ args)
executeKernel env m (KernelFunBody kernel) args = undefined --executeKernel' env kernel args

executeKernel' :: TensorEnv env -> TensorKernel env -> SArgs env () -> IO (TensorValues t)
executeKernel' env (TensorConstant sh st _ s _) args = undefined
executeKernel' _ _ _ = undefined

-- 

evalArrayInstr :: TensorEnv env -> EvalArrayInstr (ArrayInstr env)
evalArrayInstr env = EvalArrayInstr $ \instr arg -> case instr of
  Index buffer -> indexBuffer (groundRelt $ varType buffer) (toBuffer (prj' (varIdx buffer) env)) arg
  Parameter (Var tp idx) -> prjGroundVar (Var (GroundRscalar tp) idx) env

prjGroundVar :: GroundVar env t -> TensorEnv env -> t
prjGroundVar (Var _ idx) env = undefined -- ?

toBuffer :: TensorValue (Buffer t) -> Buffer t
toBuffer _ = undefined -- ?

push :: TensorEnv env -> (LeftHandSide s t env env', TensorValues t) -> TensorEnv env'
push env (LeftHandSideWildcard _, _)            = env
push env (LeftHandSideSingle _  , TupRsingle a) = env `Push` a
push env (LeftHandSidePair l1 l2, TupRpair a b) = push env (l1, a) `push` (l2, b)
push _ _                                        = error "Tuple mismatch"

type TensorEnv = Env TensorValue

data TensorValue a where
  TensorScalar :: a -> TensorValue a
  TensorBuild :: IORef (TF.Tensor TF.Build a) -> TensorValue (Buffer a)
  TensorBuild' :: TF.Tensor TF.Build a -> TensorValue (Buffer a) -- hulp nodig met IORef
  TensorValue :: IORef (TF.Tensor TF.Value a) -> TensorValue (Buffer a)
  -- misschien undefined omdat bij alloc je een placeholder nodig hebt (TF.placeholder?)

type TensorValues = TupR TensorValue