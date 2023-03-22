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
import Data.Array.Accelerate.Interpreter hiding (Right, Left)

import qualified TensorFlow.Ops                                     as TF
import qualified TensorFlow.Core                                    as TF
import qualified TensorFlow.Tensor                                  as TF
import qualified TensorFlow.Types                                   as TF
import qualified TensorFlow.Session                                 as TF
import Data.Vector (Vector)
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
import qualified Data.Vector.Storable as S

-- instance Execute SequentialSchedule TensorKernel where
--   executeAfunSchedule = undefined

type family TensorValuesIOFun t where
  TensorValuesIOFun (a -> b) = TensorValues a -> TensorValuesIOFun b
  TensorValuesIOFun t        = IO (TensorValues t) -- mischien moet t nog anders

executeSequentialSchedule :: TensorEnv env -> SequentialSchedule TensorKernel env t -> TensorValuesIOFun t
-- executeSequentialSchedule env (SequentialLam lhs sched) = executeSequentialSchedule (_ env) sched -- mist nog input ivm lambda?
-- executeSequentialSchedule env (SequentialBody sched)    = do -- run session here
--   _ <- executeSeqSchedule env sched
--   return () -- hoe moet ik van TensorValues t naar ()?
executeSequentialSchedule = undefined

executeSeqSchedule :: TensorEnv env -> SeqSchedule TensorKernel env t -> TF.Session (TensorValues t)
executeSeqSchedule env (Exec m fun args) = executeKernel env m fun args

executeSeqSchedule env (Return tr) = return $ prjAll env tr -- mapTupR
  where prjAll :: TensorEnv env -> GroundVars env t -> TensorValues t
        prjAll _ TupRunit = TupRunit
        prjAll env (TupRsingle (Var _ idx)) = TupRsingle (prj' idx env)
        prjAll env (TupRpair v v') = TupRpair (prjAll env v) (prjAll env v')

executeSeqSchedule env (Compute expr) =
  return (TupRsingle (TVScalar (evalExp expr (evalArrayInstr env))))

executeSeqSchedule env (Alet lhs _ sched sched') = do
  rhs <- executeSeqSchedule env sched
  let env' = push env (lhs, rhs)
  executeSeqSchedule env' sched'

executeSeqSchedule _ (Alloc shR st vars) = undefined -- return $ TupRsingle $ TensorValue $ liftIO (TF.placeholder _) -- ?

executeSeqSchedule _ (Use st n buffer) = do 
  x <- fromBuffer dim1 st ((), n) buffer
  ref <- liftIO $ newIORef $ Right x
  return $ TupRsingle $ TVTensor ref

executeSeqSchedule _ (Unit var) = undefined

executeSeqSchedule _ (Acond var ss ss') = undefined

executeSeqSchedule _ (Awhile tr ssf ssf' tr') = undefined


executeKernel :: TensorEnv env -> NoKernelMetadata f -> KernelFun TensorKernel args -> SArgs env args -> TF.Session (TensorValues t)
executeKernel env m (KernelFunLam z kernel) args = undefined --executeKernel (_ env) m kernel (_ args)
executeKernel env m (KernelFunBody kernel) args = undefined --executeKernel' env kernel args

-- executeKernel' :: TensorEnv env -> TensorKernel env -> SArgs env () -> IO (TensorValues t)
-- executeKernel' env (TensorConstant sh st _ s _) args = do
--   x <- newIORef $ Left $ TF.constant (TF.Shape [0]) _ -- ioref?
--   return $ TupRsingle $ TensorBuild $ x
--    --  return $ TupRsingle $ TensorBuild $ Left $ TF.constant _ _ -- ioref?
-- executeKernel' _ _ _ = undefined

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
  TVScalar :: a -> TensorValue a
  TVTensor :: IORef (Either (TF.Tensor TF.Build a) (S.Vector a)) -> TensorValue (Buffer a)
  -- misschien undefined omdat bij alloc je een placeholder nodig hebt (TF.placeholder?)

type TensorValues = TupR TensorValue